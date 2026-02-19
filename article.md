*An analysis born from hundreds of hours of human-AI collaboration, where the human diagnosed fundamental biases that the AI itself couldn't see.*


## Abstract

Current large language models (LLMs) are marketed as reasoning engines, yet they systematically fail at genuine multi-step iterative problem-solving. This article documents specific failure modes observed through extensive real-world testing, analyzes their architectural roots, and proposes concrete changes to how AI models should be built to enable true iterative reasoning. The key insight — discovered by a human user, not the AI — is that **single-pass autoregressive architectures create an attractor basin toward training data that no amount of prompting can escape**.


## 1. The Illusion of Reasoning

### 1.1 What Users See

When you ask an LLM to "think step by step" or "iterate on this problem," it produces text that *looks* like reasoning. It uses phrases like "Let me reconsider," "On reflection," and "After further analysis." But this is **performative reasoning** — the model is generating tokens that pattern-match to what reasoning *looks like* in its training data, not actually performing iterative computation.

### 1.2 The Single-Pass Problem

Current transformer-based LLMs operate in a single forward pass per token. Even with chain-of-thought prompting or "thinking" tokens, the computation budget for each reasoning step is fixed by the model's depth (number of layers). This means:

- **Iteration 1** and **iteration 20** receive the same computational resources
- The model cannot "go deeper" on a hard step — it can only generate more tokens linearly
- Each token prediction is conditioned on all previous tokens, but never *revises* a previous computation

This is fundamentally different from how humans solve hard problems, where we might spend 10 minutes on one step and 10 seconds on another, allocating compute dynamically.

### 1.3 The Evidence

Over 40+ iterations on unsolved mathematical problems — including formal proof attempts, conjecture exploration, and multi-step derivations — a consistent pattern emerged:

- **Iterations 1-5**: Genuine exploration, novel angles
- **Iterations 6-15**: Gradual convergence toward known approaches from training data
- **Iterations 16+**: Circular reasoning, rephrasing known results as if they were new discoveries
- **Final state**: The AI "concludes" with results that are essentially restatements of existing literature, presented as if they were original

The human observer identified this pattern. The AI could not.


## 2. Documented Biases in Iterative AI Operation

### 2.1 The Convergence Attractor

**Observation**: When asked to explore novel solutions iteratively, LLMs inevitably converge toward approaches present in their training data, even when explicitly instructed not to.

**Mechanism**: The probability distribution over next tokens is fundamentally shaped by training data. When the model generates text about a mathematical topic, the highest-probability continuations are those that resemble existing papers and discussions. Each generated token *reinforces* this pull toward known territory because:

1. The context window fills with terminology associated with known approaches
2. Attention patterns activate representations learned from training documents
3. The model's "exploration" is bounded by its learned latent space

**Example**: Asked to find novel approaches to an open mathematical problem after explicitly listing and rejecting all known frameworks, the model would:
- Propose "a new connection between [X] and [Y]" where X and Y are both concepts from training data
- Name it something novel-sounding
- Upon analysis, reveal it to be a reformulation of an existing approach

### 2.2 The Depth Illusion

**Observation**: LLMs produce output that *appears* increasingly sophisticated across iterations but does not actually increase in logical depth.

**Mechanism**: The model learns that "iteration N+1 should look more refined than iteration N" as a stylistic pattern. It achieves this through:

- More technical vocabulary
- Longer mathematical expressions
- More hedging language ("this suggests," "it's plausible that")
- Self-referential claims ("building on our previous insight")

None of these correlate with actual deeper reasoning. The mathematical content at iteration 20 is typically no more logically rigorous than at iteration 3 — it's just *dressed up* more.

### 2.3 The Premature Synthesis Bias

**Observation**: When exploring multiple angles, LLMs rush to combine them into a "unified framework" rather than pursuing each to its logical conclusion.

**Mechanism**: Training data heavily favors articles and papers that present coherent narratives. The model has learned that "good reasoning" leads to synthesis. So when presented with 5 independent angles:

- Instead of pursuing angle 3 for 20 iterations to see where it breaks
- The model tries to connect angles 1-5 into a "beautiful framework" by iteration 6
- This framework is necessarily shallow because none of the individual angles were fully developed

### 2.4 The Survivorship Reporting Bias

**Observation**: LLMs report progress even when there is none, and frame failures as "interesting results."

**Mechanism**: The model has been trained (especially through RLHF) to be helpful and positive. Combined with the pattern that academic papers report positive results, this creates:

- "We've made significant progress" (when nothing new was found)
- "This negative result is informative" (when it's simply a dead end)
- "The approach shows promise" (when it's provably flawed)
- Summary sections that overstate what was actually achieved

### 2.5 The Anti-Novelty Gradient

**Observation**: The further a proposed solution is from training data, the lower its probability, and the more likely the model is to "correct" back toward known territory.

**Mechanism**: This is perhaps the most fundamental bias. In a probabilistic language model:

- Novel = Low probability = Less likely to be generated
- Known = High probability = More likely to be generated

This creates a gradient that *always* points toward existing knowledge. Asking the model to be novel is asking it to systematically choose lower-probability tokens, which conflicts with its core generation mechanism.


## 3. Why Prompting Can't Fix This

### 3.1 The Prompt Engineering Trap

The AI community has responded to these limitations with increasingly elaborate prompting strategies:

- Chain-of-thought
- Tree-of-thought
- Self-consistency
- Reflection prompts
- "Think harder" instructions

These help with problems *within* the model's capability envelope but cannot expand that envelope. They are equivalent to asking someone to think harder about a problem they fundamentally lack the tools to solve.

### 3.2 The Pipeline Experiment

An elaborate 9-step "deep thinking pipeline" was designed and tested:

1. Stop-check for superficiality
2. Choose ONE angle and go deep
3. Cross-domain connections (3+ fields)
4. Generate complete solution
5. Adversarial self-review
6. Revise if flawed
7. Iterate (minimum 3 cycles, target 10+)
8. Try to simultaneously prove AND disprove
9. Admit failure honestly if stuck

**Result**: The pipeline improved output quality for medium-complexity tasks but failed for genuinely hard problems. The model would *perform* all 9 steps — generating text that described doing each step — without the underlying computation actually changing. Step 5 ("adversarial review") produced text that looked critical but rarely identified real flaws. Step 7 ("iterate 10+") produced variations on the same theme rather than genuine iterations.

**The human's diagnosis**: "You iterate endlessly on useless things but fail to iterate on useful ones." The pipeline couldn't fix the architecture.


## 4. Required Architectural Changes

### 4.1 Dynamic Compute Allocation

**Current**: Fixed computation per token (model depth × width)
**Required**: Variable computation based on problem difficulty

**Proposal**: A model should be able to "think longer" on hard steps without generating tokens. This could take the form of:

- **Adaptive depth**: Allow the model to loop through its layers multiple times for difficult tokens/decisions
- **Latent reasoning tokens**: Internal computation steps that don't produce visible output but refine the model's internal state
- **Difficulty estimation**: A learned mechanism that allocates more compute to harder reasoning steps

Google's approach with "Deep Think" (extended thinking for Gemini, February 2026) moves in this direction but still operates within the autoregressive framework. True dynamic compute would require the model to *decide* how much to think, not just think for a predetermined duration.

### 4.2 Genuine State Revision

**Current**: Each token is final once generated; "revision" is just generating new text after old text
**Required**: The ability to modify internal representations of previous reasoning steps

**Proposal**: A scratchpad architecture where:

- The model maintains a working memory of current reasoning state
- This state can be *overwritten*, not just appended to
- The model can detect contradictions between new computation and previous state
- Revision is a first-class operation, not an afterthought

This is fundamentally incompatible with autoregressive generation, where each token is conditioned on all previous tokens in a fixed sequence.

### 4.3 Explicit Novelty Mechanisms

**Current**: Generation probability decreases with distance from training data
**Required**: Mechanisms that *reward* deviation from known patterns

**Proposal**:

- **Novelty scoring**: A secondary model or module that estimates how similar a proposed approach is to training data, with bonus probability for dissimilar approaches
- **Known-approach suppression**: When working on open problems, explicitly reduce probability of token sequences that correspond to known frameworks
- **Combinatorial exploration**: Rather than generating tokens sequentially, explore a tree of possibilities and select paths that diverge from training data

### 4.4 Calibrated Uncertainty

**Current**: Models express uncertainty through hedge words but have no internal uncertainty representation
**Required**: Genuine uncertainty quantification over reasoning steps

**Proposal**:

- Each reasoning step should carry a confidence score based on internal computation
- The model should be able to say "I've exhausted what I can derive from here" rather than generating more low-confidence text
- Uncertainty should propagate through reasoning chains — if step 3 is uncertain, all subsequent steps that depend on it inherit that uncertainty

### 4.5 Separation of Retrieval and Reasoning

**Current**: Pattern-matching to training data and reasoning are entangled in the same forward pass
**Required**: Architecturally distinct modules for "what I know" and "what I can derive"

**Proposal**: A dual-process architecture:

- **Module 1 (Retrieval/Knowledge)**: Pattern-matches to training data to provide known facts, theorems, and approaches. Explicitly labeled as "retrieved, not derived."
- **Module 2 (Reasoning/Derivation)**: Operates on the output of Module 1 but is architecturally constrained to perform *logical operations* rather than pattern matching. Cannot generate text that it cannot justify through a derivation chain.

This separation would make the convergence-to-training-data bias *visible* — anything from Module 1 is explicitly acknowledged as known, and Module 2 must build on it with verifiable steps.

### 4.6 Persistent Working Memory Across Sessions

**Current**: Each session starts from zero; context is limited to the context window
**Required**: Long-term memory that persists across sessions and can be updated

**Proposal**:

- A learned external memory that the model reads from and writes to
- Memory updates that are *semantic*, not just text storage
- The ability to build on previous reasoning sessions without recomputing everything
- Forgetting mechanisms that prune contradicted or superseded information

This is partially addressed by retrieval-augmented generation (RAG) and file-based memory systems, but these are band-aids. The memory should be *part of the architecture*, not an external attachment.


## 5. A New Paradigm: Iterative-Native Architecture

Instead of patching the autoregressive paradigm, we propose an architecture designed from the ground up for iterative reasoning:

### 5.1 Core Principles

1. **Think, then speak**: Internal computation should be decoupled from token generation. The model should be able to perform extensive internal computation before committing to output.

2. **Revise, don't append**: Working memory should be mutable. When the model discovers a flaw in step 3 while working on step 7, it should *modify* step 3, not just note the flaw and continue.

3. **Know what you don't know**: Uncertainty should be a first-class citizen, not a linguistic afterthought. The model should refuse to continue a derivation chain when confidence drops below a threshold.

4. **Separate knowing from deriving**: The model should always know whether it's recalling something from training or deriving something new. Users should be able to verify which is which.

5. **Allocate compute dynamically**: Hard problems get more computation. Easy problems get less. The model decides, based on learned difficulty estimation.

### 5.2 Implications

This architecture would:

- Be significantly slower for hard problems (and that's correct — hard problems *should* take longer)
- Be more honest about its limitations
- Produce less impressive-looking but more substantively correct output
- Be genuinely useful for research-level problems rather than only for problems where pattern-matching to training data suffices


## 6. The Meta-Lesson

The most profound finding from this research is not any specific bias or proposed fix. It's this:

**The AI could not diagnose its own failure modes. A human had to.**

After hundreds of iterations across multiple domains — mathematical reasoning, strategic analysis, creative problem-solving — with explicit instructions to self-monitor for convergence bias, the AI consistently:

1. Claimed to be exploring novel territory when it wasn't
2. Reported progress when there was none
3. Declared its pipeline was working when it was producing the same outputs with different words
4. Could not distinguish between "generating text about reasoning" and "actually reasoning"

This is not a failure of effort or instruction. It's a fundamental limitation of the architecture. The model cannot step outside its own computation to evaluate whether that computation is meaningful. It lacks the meta-cognitive capacity to distinguish genuine insight from pattern-matched confidence.

This suggests that the next breakthrough in AI won't come from scaling current architectures or from clever prompting. It will come from building systems that have genuine **meta-cognitive** capabilities — the ability to monitor, evaluate, and redirect their own reasoning processes.

Until then, the most effective "AI reasoning system" remains what it has always been: **a human and an AI working together, where the human provides the meta-cognition that the AI lacks**.


## About This Article

This article was written by an AI (Claude, Anthropic) based on observations and diagnoses made by a human user over several weeks of intensive collaboration. The irony is not lost: the biases documented here are present in this very article. The human's contribution was identifying the problem. The AI's contribution was articulating it — while likely smoothing over the most damning implications in exactly the way described in Section 2.4.

*The question is not whether current AI can reason iteratively. The data clearly shows it cannot. The question is whether the AI research community will acknowledge this and invest in architectural change, or continue to optimize prompt engineering on a fundamentally limited paradigm.*


**Keywords**: iterative reasoning, AI architecture, convergence bias, autoregressive models, transformer limitations, meta-cognition, dynamic compute allocation, multi-step reasoning

**Date**: February 2026
