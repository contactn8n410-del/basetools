/**
 * BaseTools API — Google Apps Script Backend
 * Deployed as web app, serves as API for the frontend
 */

const RPC_URL = 'https://base-rpc.publicnode.com';
const TREASURY = '0x0282BdE2f138babC6ABa3bb010121112cC1d7eDa';
const PREMIUM_FEE_WEI = '1000000000000000'; // 0.001 ETH
const FREE_LIMIT = 3;

// Rate limiting via Properties
function getRateKey(ip) {
  return 'rate_' + ip + '_' + new Date().toISOString().slice(0, 10);
}

function doGet(e) {
  return handleRequest(e);
}

function doPost(e) {
  return handleRequest(e);
}

function handleRequest(e) {
  const action = e.parameter.action;
  const addr = (e.parameter.address || '').toLowerCase();
  
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Content-Type': 'application/json'
  };
  
  try {
    let result;
    switch(action) {
      case 'analyze':
        result = analyzeWallet(addr);
        break;
      case 'verify_payment':
        result = verifyPayment(e.parameter.txHash, e.parameter.userAddr);
        break;
      case 'deep_audit':
        result = deepAudit(addr, e.parameter.txHash);
        break;
      default:
        result = {error: 'Unknown action'};
    }
    
    return ContentService.createTextOutput(JSON.stringify(result))
      .setMimeType(ContentService.MimeType.JSON);
  } catch(err) {
    return ContentService.createTextOutput(JSON.stringify({error: err.message}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function rpcCall(method, params) {
  const resp = UrlFetchApp.fetch(RPC_URL, {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify({jsonrpc: '2.0', id: 1, method: method, params: params}),
    muteHttpExceptions: true
  });
  return JSON.parse(resp.getContentText()).result;
}

function analyzeWallet(addr) {
  if (!/^0x[0-9a-f]{40}$/.test(addr)) return {error: 'Invalid address'};
  
  const [code, balance, txCount] = [
    rpcCall('eth_getCode', [addr, 'latest']),
    rpcCall('eth_getBalance', [addr, 'latest']),
    rpcCall('eth_getTransactionCount', [addr, 'latest'])
  ];
  
  const isContract = code && code !== '0x';
  const balEth = parseInt(balance, 16) / 1e18;
  const nonce = parseInt(txCount, 16);
  
  let analysis = {
    address: addr,
    type: isContract ? 'contract' : 'wallet',
    balance_eth: balEth,
    balance_usd: balEth * 2700, // approximate
    tx_count: nonce,
    code_size: isContract ? (code.length - 2) / 2 : 0
  };
  
  if (isContract) {
    analysis.patterns = detectPatterns(code);
    analysis.standards = detectStandards(code);
  }
  
  // Get recent transactions via eth_getLogs for contracts
  if (isContract) {
    try {
      const latestBlock = rpcCall('eth_blockNumber', []);
      const fromBlock = '0x' + Math.max(0, parseInt(latestBlock, 16) - 10000).toString(16);
      const logs = rpcCall('eth_getLogs', [{
        address: addr,
        fromBlock: fromBlock,
        toBlock: 'latest'
      }]);
      analysis.recent_events = (logs || []).length;
    } catch(e) {
      analysis.recent_events = -1;
    }
  }
  
  return analysis;
}

function detectPatterns(code) {
  const patterns = [];
  if (code.includes('a9059cbb')) patterns.push('transfer');
  if (code.includes('095ea7b3')) patterns.push('approve');
  if (code.includes('23b872dd')) patterns.push('transferFrom');
  if (code.includes('3659cfe6')) patterns.push('upgradeTo (PROXY)');
  if (code.includes('f851a440')) patterns.push('admin (PROXY)');
  if (code.includes('5c60da1b')) patterns.push('implementation (PROXY)');
  if (code.includes('3d3d3d3d')) patterns.push('MINIMAL_PROXY');
  return patterns;
}

function detectStandards(code) {
  const standards = [];
  if (code.includes('a9059cbb') && code.includes('095ea7b3') && code.includes('18160ddd')) 
    standards.push('ERC20');
  if (code.includes('42842e0e')) standards.push('ERC721');
  if (code.includes('f242432a')) standards.push('ERC1155');
  return standards;
}

function verifyPayment(txHash, userAddr) {
  if (!/^0x[0-9a-f]{64}$/.test(txHash)) return {valid: false, error: 'Invalid tx hash'};
  
  const tx = rpcCall('eth_getTransactionByHash', [txHash]);
  if (!tx) return {valid: false, error: 'Transaction not found'};
  
  const receipt = rpcCall('eth_getTransactionReceipt', [txHash]);
  if (!receipt || receipt.status !== '0x1') return {valid: false, error: 'Transaction failed'};
  
  // Check recipient is treasury
  if (tx.to?.toLowerCase() !== TREASURY.toLowerCase()) 
    return {valid: false, error: 'Wrong recipient'};
  
  // Check amount
  const value = BigInt(tx.value);
  if (value < BigInt(PREMIUM_FEE_WEI))
    return {valid: false, error: 'Insufficient amount'};
  
  // Store verified payment
  const props = PropertiesService.getScriptProperties();
  const payments = JSON.parse(props.getProperty('payments') || '{}');
  payments[txHash] = {
    from: tx.from,
    value: tx.value,
    timestamp: new Date().toISOString(),
    used: false
  };
  props.setProperty('payments', JSON.stringify(payments));
  
  return {valid: true, from: tx.from};
}

function deepAudit(addr, txHash) {
  // Verify payment first
  const payment = verifyPayment(txHash, addr);
  if (!payment.valid) return {error: 'Payment not verified: ' + payment.error};
  
  // Mark payment as used
  const props = PropertiesService.getScriptProperties();
  const payments = JSON.parse(props.getProperty('payments') || '{}');
  if (payments[txHash]) {
    payments[txHash].used = true;
    payments[txHash].audit_address = addr;
    props.setProperty('payments', JSON.stringify(payments));
  }
  
  // Full analysis
  const basic = analyzeWallet(addr);
  if (basic.error) return basic;
  
  const code = rpcCall('eth_getCode', [addr, 'latest']);
  
  // Deep bytecode analysis
  let riskScore = 0;
  let findings = [];
  
  // Dangerous opcodes
  if (code.includes('ff')) { findings.push({sev:'HIGH', msg:'SELFDESTRUCT — contract can be destroyed'}); riskScore+=30; }
  if (code.includes('f4')) { findings.push({sev:'HIGH', msg:'DELEGATECALL — external code execution in own context'}); riskScore+=25; }
  if (code.includes('3659cfe6')) { findings.push({sev:'HIGH', msg:'Proxy pattern (upgradeTo) — implementation can change'}); riskScore+=20; }
  if (code.includes('f0')) { findings.push({sev:'MEDIUM', msg:'CREATE opcode — can deploy child contracts'}); riskScore+=10; }
  if (code.includes('f5')) { findings.push({sev:'MEDIUM', msg:'CREATE2 — deterministic deployment'}); riskScore+=10; }
  
  // Reentrancy indicators
  const callCount = (code.match(/f1/g) || []).length;
  if (callCount > 5) { findings.push({sev:'MEDIUM', msg:`Multiple CALL opcodes (${callCount}) — check reentrancy guards`}); riskScore+=15; }
  
  // Storage analysis
  const storageSlots = [];
  for (let i = 0; i < 10; i++) {
    const slot = rpcCall('eth_getStorageAt', [addr, '0x' + i.toString(16), 'latest']);
    if (slot && slot !== '0x0000000000000000000000000000000000000000000000000000000000000000') {
      storageSlots.push({slot: i, value: slot});
    }
  }
  
  // Extract selectors
  const selectors = new Set();
  for (let i = 0; i < code.length - 10; i++) {
    if (code.substr(i, 2) === '63') { // PUSH4
      selectors.add('0x' + code.substr(i+2, 8));
    }
  }
  
  return {
    ...basic,
    risk_score: Math.min(riskScore, 100),
    risk_level: riskScore < 20 ? 'LOW' : riskScore < 50 ? 'MEDIUM' : 'HIGH',
    findings: findings,
    storage_slots: storageSlots,
    selector_count: selectors.size,
    selectors: Array.from(selectors).slice(0, 30),
    audit_timestamp: new Date().toISOString()
  };
}

// Revenue tracking
function getRevenue() {
  const props = PropertiesService.getScriptProperties();
  const payments = JSON.parse(props.getProperty('payments') || '{}');
  let total = 0;
  let count = 0;
  for (const [hash, data] of Object.entries(payments)) {
    total += parseInt(data.value, 16) / 1e18;
    count++;
  }
  return {total_eth: total, payment_count: count};
}
