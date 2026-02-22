#!/usr/bin/env python3
"""MDIR — Multi-Depth Iterative Reasoning — Training Script for Colab T4"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
import math, time, json, os, shutil

# ============================================================
# Google Drive for persistent checkpoints
# ============================================================
DRIVE_DIR = '/content/drive/MyDrive/mdir_checkpoints'
if os.path.isdir('/content/drive/MyDrive'):
    # Drive already mounted (e.g. from notebook cell)
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print(f'✅ Google Drive already mounted — checkpoints → {DRIVE_DIR}')
    USE_DRIVE = True
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        os.makedirs(DRIVE_DIR, exist_ok=True)
        print(f'✅ Google Drive mounted — checkpoints → {DRIVE_DIR}')
        USE_DRIVE = True
    except Exception as e:
        print(f'⚠️ Google Drive not available ({e}) — checkpoints local only')
        DRIVE_DIR = '/content/checkpoints'
        USE_DRIVE = False

# ============================================================
# Device
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ============================================================
# Config
# ============================================================
@dataclass
class MDIRConfig:
    backbone_dim: int = 256
    backbone_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 256
    tap_layers: tuple = (2, 4, 6, 8)
    rh_dim: int = 192
    rh_layers: int = 3
    rh_heads: int = 6
    n_roles: int = 4
    max_iterations: int = 8
    assembler_layers: int = 4
    diversity_lambda: float = 0.3
    role_lambda: float = 0.2
    dropout: float = 0.1

config = MDIRConfig()
print(f'Config: {config}')

# ============================================================
# Building Blocks
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(x, freqs):
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        freqs = self.rotary(S).to(x.device)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        return self.o(out.transpose(1, 2).contiguous().view(B, S, D))

class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        hidden = dim * 4
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, dropout)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

print('Building blocks ✅')

# ============================================================
# MDIR Components
# ============================================================
class QueryTranslator(nn.Module):
    def __init__(self, backbone_dim, rh_dim, depth_index):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, rh_dim * 2), nn.GELU(),
            nn.Linear(rh_dim * 2, rh_dim), RMSNorm(rh_dim)
        )
        self.depth_embed = nn.Parameter(torch.randn(1, 1, rh_dim) * 0.02)
    def forward(self, h):
        return self.proj(h) + self.depth_embed

class ReasoningHead(nn.Module):
    def __init__(self, dim, n_layers, n_heads, vocab_size, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.conf_head = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1), nn.Sigmoid())
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        h = self.norm(x)
        return self.lm_head(h), self.conf_head(h.mean(dim=1)), h

class RoleModulator(nn.Module):
    def __init__(self, dim, n_roles=4):
        super().__init__()
        self.transforms = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.Tanh()) for _ in range(n_roles)])
        self.scales = nn.Parameter(torch.ones(n_roles))
    def forward(self, hidden, role_onehot):
        transformed = torch.stack([t(hidden) * self.scales[i] for i, t in enumerate(self.transforms)], dim=0)
        weights = role_onehot.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        return (transformed * weights).sum(dim=0)

class DynamicRouter(nn.Module):
    def __init__(self, input_dim, n_rhs, n_roles=4, max_iters=8):
        super().__init__()
        hidden = input_dim // 2
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), RMSNorm(hidden))
        self.role_head = nn.Linear(hidden, n_rhs * n_roles)
        self.priority_head = nn.Linear(hidden, n_rhs)
        self.iter_head = nn.Linear(hidden, max_iters)
        self.gru = nn.GRUCell(hidden + n_rhs, hidden)
        self.n_rhs = n_rhs
        self.n_roles = n_roles
    def forward(self, query_embed, prev_conf=None, prev_state=None, tau=0.5):
        h = self.encoder(query_embed)
        if prev_conf is not None and prev_state is not None:
            h = self.gru(torch.cat([h, prev_conf], -1), prev_state)
        role_logits = self.role_head(h).view(-1, self.n_rhs, self.n_roles)
        if self.training:
            roles = F.gumbel_softmax(role_logits, tau=tau, hard=True)
        else:
            roles = F.one_hot(role_logits.argmax(-1), self.n_roles).float()
        return {
            'roles': roles,
            'priorities': torch.sigmoid(self.priority_head(h)),
            'n_iterations': (self.iter_head(h).argmax(-1) + 1).clamp(max=8),
            'state': h
        }

class WorkingMemory:
    def __init__(self):
        self.slots = []
    def write(self, iteration, rh_id, hidden, confidence):
        self.slots.append({'iter': iteration, 'rh': rh_id, 'hidden': hidden.detach(), 'conf': confidence})
    def read_for_rh(self, rh_id, current_iter):
        relevant = [s for s in self.slots if s['rh'] != rh_id and s['iter'] == current_iter - 1]
        if not relevant:
            return None
        total = sum(s['conf'] for s in relevant)
        if total == 0:
            return None
        return sum(s['hidden'] * (s['conf'] / total) for s in relevant)
    def clear(self):
        self.slots.clear()

class Assembler(nn.Module):
    def __init__(self, dim, n_rhs, vocab_size, n_layers=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.cross_norm = RMSNorm(dim)
        self.agreement = nn.Sequential(nn.Linear(dim * n_rhs, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim), RMSNorm(dim))
        self.gen_layers = nn.ModuleList([TransformerBlock(dim, 8) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.meta_conf = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 3))
    def forward(self, rh_hiddens, rh_confs, query_hidden):
        weighted = [h * rh_confs[:, i:i+1].unsqueeze(-1) for i, h in enumerate(rh_hiddens)]
        all_kv = torch.cat(weighted, dim=1)
        x = query_hidden
        residual = x
        x, attn_w = self.cross_attn(self.cross_norm(x), all_kv, all_kv)
        x = x + residual
        agreement = self.agreement(torch.cat([h.mean(1) for h in rh_hiddens], -1))
        x = x + agreement.unsqueeze(1).expand_as(x)
        for layer in self.gen_layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x), self.meta_conf(x.mean(1)), attn_w

print('MDIR components ✅')

# ============================================================
# Full MDIR Model
# ============================================================
class MDIRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_taps = len(config.tap_layers)
        self.embed = nn.Embedding(config.vocab_size, config.backbone_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.backbone_dim)
        self.backbone = nn.ModuleList([TransformerBlock(config.backbone_dim, config.n_heads, config.dropout) for _ in range(config.backbone_layers)])
        self.qts = nn.ModuleList([QueryTranslator(config.backbone_dim, config.rh_dim, i) for i in range(n_taps)])
        self.rhs = nn.ModuleList([ReasoningHead(config.rh_dim, config.rh_layers, config.rh_heads, config.vocab_size, config.dropout) for _ in range(n_taps)])
        self.role_mods = nn.ModuleList([RoleModulator(config.rh_dim, config.n_roles) for _ in range(n_taps)])
        self.router = DynamicRouter(config.backbone_dim, n_taps, config.n_roles, config.max_iterations)
        self.assembler = Assembler(config.rh_dim, n_taps, config.vocab_size, config.assembler_layers)
        self.wm_proj = nn.Linear(config.rh_dim, config.backbone_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

    def extract_taps(self, input_ids, extra_ctx=None):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        if extra_ctx is not None:
            x = torch.cat([x, extra_ctx], dim=1)
        total_len = x.size(1)
        mask = torch.tril(torch.ones(total_len, total_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        tapped = []
        tap_idx = 0
        for i, layer in enumerate(self.backbone):
            x = layer(x, mask)
            if tap_idx < len(self.config.tap_layers) and (i + 1) == self.config.tap_layers[tap_idx]:
                tapped.append(x.clone())
                tap_idx += 1
        return tapped

    def forward(self, input_ids, targets=None, n_iters=2):
        B, S = input_ids.shape
        n_taps = len(self.config.tap_layers)
        tapped = self.extract_taps(input_ids)
        query_embed = tapped[-1].mean(dim=1)
        routing = self.router(query_embed)
        wm = WorkingMemory()
        all_rh_logits = [[] for _ in range(n_taps)]

        for t in range(n_iters):
            rh_hiddens = []
            rh_confs = []
            for i in range(n_taps):
                rh_in = self.qts[i](tapped[i])
                wm_ctx = wm.read_for_rh(i, t)
                if wm_ctx is not None:
                    rh_in = torch.cat([rh_in, wm_ctx], dim=1)
                rh_in = self.role_mods[i](rh_in, routing['roles'][:, i])
                logits, conf, hidden = self.rhs[i](rh_in)
                rh_hiddens.append(hidden)
                rh_confs.append(conf.squeeze(-1))
                all_rh_logits[i].append(logits)
                wm.write(t, i, hidden, conf.mean().item())
            confs_t = torch.stack(rh_confs, -1)
            routing = self.router(query_embed, confs_t, routing['state'])
            if t < n_iters - 1:
                summary = torch.cat([h.mean(1, keepdim=True) for h in rh_hiddens], 1)
                extra = self.wm_proj(summary)
                tapped = self.extract_taps(input_ids, extra)

        final_confs = torch.stack(rh_confs, -1)
        query_rh = self.qts[-1](tapped[-1])
        final_logits, meta_conf, attn_w = self.assembler(rh_hiddens, final_confs, query_rh)

        result = {
            'logits': final_logits,
            'meta_confidence': meta_conf,
            'rh_confidences': final_confs,
            'n_iterations': n_iters,
        }
        if targets is not None:
            result['loss'], result['loss_parts'] = self._loss(final_logits, all_rh_logits, routing, final_confs, targets)
        return result

    def _loss(self, final_logits, all_rh_logits, routing, confs, targets):
        cfg = self.config
        sl = min(final_logits.size(1), targets.size(1))
        asm = F.cross_entropy(final_logits[:, :sl].reshape(-1, final_logits.size(-1)), targets[:, :sl].reshape(-1), ignore_index=-100)
        task = 0; cnt = 0
        for rh_list in all_rh_logits:
            for lg in rh_list:
                s = min(lg.size(1), targets.size(1))
                task += F.cross_entropy(lg[:, :s].reshape(-1, lg.size(-1)), targets[:, :s].reshape(-1), ignore_index=-100)
                cnt += 1
        task /= max(cnt, 1)
        div = torch.tensor(0.0, device=final_logits.device)
        last = [rh[-1] for rh in all_rh_logits]
        pairs = 0
        for i in range(len(last)):
            for j in range(i+1, len(last)):
                s = min(last[i].size(1), last[j].size(1))
                pi = F.softmax(last[i][:, :s], -1)
                pj = F.softmax(last[j][:, :s], -1)
                m = (pi + pj) / 2
                js = (F.kl_div(m.log(), pi, reduction='batchmean') + F.kl_div(m.log(), pj, reduction='batchmean')) / 2
                div -= js
                pairs += 1
        div /= max(pairs, 1)
        total = asm + task + cfg.diversity_lambda * div
        return total, {'asm': asm.item(), 'task': task.item(), 'div': div.item(), 'total': total.item()}

# ============================================================
# Build model (temporary vocab)
# ============================================================
model = MDIRModel(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'\nMDIR Model: {n_params/1e6:.1f}M parameters')

# ============================================================
# Dataset
# ============================================================
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
actual_vocab = tokenizer.vocab_size
print(f'Tokenizer vocab: {actual_vocab}')

# Rebuild with correct vocab
config.vocab_size = actual_vocab
model = MDIRModel(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'MDIR Model (updated vocab): {n_params/1e6:.1f}M parameters')
print(f'Estimated VRAM: ~{n_params * 4 / 1e9 * 3:.1f} GB')

print('Loading TinyStories...')
ds = load_dataset('roneneldan/TinyStories', split='train[:50000]')
val_ds = load_dataset('roneneldan/TinyStories', split='validation[:2000]')
print(f'Train: {len(ds)} | Val: {len(val_ds)}')

def tokenize_fn(examples):
    tokens = tokenizer(examples['text'], truncation=True, max_length=config.max_seq_len, padding='max_length', return_tensors='pt')
    tokens['labels'] = tokens['input_ids'].clone()
    tokens['labels'][tokens['attention_mask'] == 0] = -100
    return tokens

print('Tokenizing...')
train_tok = ds.map(tokenize_fn, batched=True, batch_size=256, remove_columns=ds.column_names)
val_tok = val_ds.map(tokenize_fn, batched=True, batch_size=256, remove_columns=val_ds.column_names)
train_tok.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_tok.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
train_loader = DataLoader(train_tok, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_tok, batch_size=4, num_workers=2, pin_memory=True)
print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

# ============================================================
# Training
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
total_steps = len(train_loader) * 3
warmup_steps = total_steps // 10

def lr_schedule(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

phase_config = [
    {'epoch': 1, 'n_iters': 1, 'div_lambda': 0.0, 'desc': 'Phase 1: Learn to generate'},
    {'epoch': 2, 'n_iters': 2, 'div_lambda': 0.15, 'desc': 'Phase 2: Diversify'},
    {'epoch': 3, 'n_iters': 3, 'div_lambda': 0.3, 'desc': 'Phase 3: Full iterations'},
]

print(f'Total steps: {total_steps} | Warmup: {warmup_steps}')

scaler = torch.amp.GradScaler('cuda')
log_every = 100
save_every = 500
history = []
global_step = 0
os.makedirs('checkpoints', exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)

# === Resume from checkpoint if available ===
resume_path = os.path.join(DRIVE_DIR, 'mdir_latest.pt')
start_epoch = 1
if os.path.exists(resume_path):
    print(f'🔄 Resuming from {resume_path}...')
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    global_step = ckpt['step']
    history = ckpt.get('history', [])
    start_epoch = ckpt.get('epoch', 1)
    # Advance scheduler to correct position
    for _ in range(global_step):
        scheduler.step()
    print(f'  ✅ Resumed at step {global_step}, epoch {start_epoch}, loss {history[-1]["loss"]:.4f}')
else:
    print('🆕 No checkpoint found — training from scratch')

for phase in phase_config:
    if phase['epoch'] < start_epoch:
        print(f"⏭️ Skipping {phase['desc']} (already done)")
        continue
    epoch = phase['epoch']
    n_iters = phase['n_iters']
    model.config.diversity_lambda = phase['div_lambda']
    print(f"\n{'='*60}")
    print(f"{phase['desc']} | iters={n_iters} | div_λ={phase['div_lambda']}")
    print(f"{'='*60}")

    model.train()
    epoch_loss = 0
    epoch_steps = 0
    t0 = time.time()

    # Calculate how many batches to skip if resuming mid-epoch
    skip_batches = global_step - (phase['epoch'] - 1) * len(train_loader) if phase['epoch'] == start_epoch and global_step > 0 else 0
    if skip_batches > 0:
        print(f'  ⏭️ Skipping {skip_batches} batches (already trained)')

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx < skip_batches:
            continue
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            output = model(input_ids, targets=labels, n_iters=n_iters)
            loss = output['loss']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_steps += 1
        global_step += 1

        if global_step % log_every == 0:
            avg_loss = epoch_loss / epoch_steps
            elapsed = time.time() - t0
            steps_per_sec = epoch_steps / elapsed
            lr = scheduler.get_last_lr()[0]
            parts = output['loss_parts']
            confs = output['rh_confidences'].mean(0).tolist()
            log = {'step': global_step, 'epoch': epoch, 'loss': avg_loss, 'lr': lr, 'steps/s': steps_per_sec, **parts, 'rh_confs': confs}
            history.append(log)
            print(f"  step {global_step:5d} | loss {avg_loss:.4f} | "
                  f"asm {parts['asm']:.3f} task {parts['task']:.3f} div {parts['div']:.3f} | "
                  f"confs {[f'{c:.3f}' for c in confs]} | "
                  f"{steps_per_sec:.1f} steps/s | lr {lr:.2e}", flush=True)

        if global_step % save_every == 0:
            ckpt_data = {'step': global_step, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'history': history}
            local_path = f'checkpoints/mdir_step{global_step}.pt'
            torch.save(ckpt_data, local_path)
            # Copy to Drive for persistence
            drive_path = os.path.join(DRIVE_DIR, 'mdir_latest.pt')
            try:
                shutil.copy2(local_path, drive_path)
                shutil.copy2(local_path, os.path.join(DRIVE_DIR, f'mdir_step{global_step}.pt'))
                print(f'  💾 Saved checkpoint at step {global_step} → Drive ✅')
            except Exception as e:
                print(f'  💾 Saved checkpoint at step {global_step} (Drive failed: {e})')

    # Validation
    model.eval()
    val_loss = 0; val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            with torch.amp.autocast('cuda'):
                output = model(input_ids, targets=labels, n_iters=n_iters)
            val_loss += output['loss'].item()
            val_steps += 1
    val_avg = val_loss / val_steps
    train_avg = epoch_loss / epoch_steps
    print(f"\n  Epoch {epoch} done | Train loss: {train_avg:.4f} | Val loss: {val_avg:.4f}")

# Save final
final_data = {'step': global_step, 'epoch': 3, 'model': model.state_dict(), 'config': config, 'history': history}
torch.save(final_data, 'checkpoints/mdir_final.pt')
try:
    shutil.copy2('checkpoints/mdir_final.pt', os.path.join(DRIVE_DIR, 'mdir_final.pt'))
    shutil.copy2('checkpoints/mdir_final.pt', os.path.join(DRIVE_DIR, 'mdir_latest.pt'))
    print('\n🎉 Training complete! Final checkpoint saved to Drive ✅')
except:
    print('\n🎉 Training complete! Final checkpoint saved locally.')

# ============================================================
# Evaluation
# ============================================================
print('\n=== Generation Test ===')
model.eval()
prompts = ["Once upon a time, there was a", "The scientist discovered that", "If you add 3 and 5, you get", "The cat sat on the"]

for prompt in prompts:
    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        for n_it in [1, 3]:
            generated = tokens.clone()
            for _ in range(50):
                if generated.size(1) >= config.max_seq_len:
                    break
                out = model(generated, n_iters=n_it)
                next_tok = out['logits'][:, -1, :].argmax(-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            conf = out['meta_confidence'].softmax(-1).tolist()[0]
            rh_c = out['rh_confidences'].tolist()[0]
            print(f'[iters={n_it}] {text}')
            print(f'  Meta-conf: {[f"{c:.2f}" for c in conf]} | RH confs: {[f"{c:.3f}" for c in rh_c]}')
    print()

# Diversity analysis
print('=== Diversity Analysis ===')
prompt = "The reason why water boils is"
tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
with torch.no_grad():
    tapped = model.extract_taps(tokens)
    query_embed = tapped[-1].mean(dim=1)
    routing = model.router(query_embed)
    roles = routing['roles'][0]
    role_names = ['Lead', 'Critic', 'Explorer', 'Verifier']
    priorities = routing['priorities'][0]
    for i in range(len(config.tap_layers)):
        role = role_names[roles[i].argmax().item()]
        prio = priorities[i].item()
        print(f'  RH-{i+1} (depth {config.tap_layers[i]}): role={role}, priority={prio:.3f}')
    print(f'  Iterations: {routing["n_iterations"].item()}')

    print(f'\nIndividual RH outputs:')
    for i in range(len(config.tap_layers)):
        rh_in = model.qts[i](tapped[i])
        rh_in = model.role_mods[i](rh_in, routing['roles'][:, i])
        logits, conf, _ = model.rhs[i](rh_in)
        top5 = logits[0, -1].topk(5)
        top5_words = [tokenizer.decode([t]) for t in top5.indices.tolist()]
        role = role_names[roles[i].argmax().item()]
        print(f'  RH-{i+1} ({role}, conf={conf.item():.3f}): next → {top5_words}')

# Save artifacts
with open('training_history.json', 'w') as f:
    json.dump(history, f, indent=2)
with open('mdir_config.json', 'w') as f:
    json.dump({k: str(v) if isinstance(v, tuple) else v for k, v in vars(config).items()}, f, indent=2)

print('\n📦 All artifacts saved!')
print('  checkpoints/mdir_final.pt')
print('  training_history.json')
print('  mdir_config.json')
