from web3 import Web3
from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

ANCHOR_CONTRACT_ADDRESS = "0xd9145CCE52D386f254917e481eB44e9943F39138"

ANALYSIS_ANCHOR_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "datasetHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "resultHash", "type": "bytes32"}
        ],
        "name": "commitReport",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "bytes32", "name": "datasetHash", "type": "bytes32"},
            {"indexed": False, "internalType": "bytes32", "name": "resultHash", "type": "bytes32"},
            {"indexed": False, "internalType": "address", "name": "submitter", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "blockNumber", "type": "uint256"}
        ],
        "name": "ReportCommitted",
        "type": "event"
    }
]

_anchor_contract = None

def _get_anchor_contract():
    global _anchor_contract
    if _anchor_contract is not None:
        return _anchor_contract
    if not ANCHOR_CONTRACT_ADDRESS:
        return None
    try:
        _anchor_contract = w3.eth.contract(
            address=Web3.to_checksum_address(ANCHOR_CONTRACT_ADDRESS),
            abi=ANALYSIS_ANCHOR_ABI
        )
        return _anchor_contract
    except Exception:
        _anchor_contract = None
        return None


def chain_info():
    ok = w3.is_connected()
    if not ok:
        return {"connected": False}
    latest = w3.eth.get_block("latest")
    return {
        "connected": True,
        "chain_id": int(w3.eth.chain_id),
        "block_number": int(latest.number),
        "latest_block_hash": latest.hash.hex(),
    }


def commit_report(dataset_hash_hex: str, result_hash_hex: str):
    """
    dataset_hash_hex/result_hash_hex: 64位hex字符串（不带0x）
    返回：{ok, tx_hash, block_number, from} 或 {ok:False, message:"..."}
    """
    if not w3.is_connected():
        return {"ok": False, "message": "RPC 未连接"}

    c = _get_anchor_contract()
    if c is None:
        return {"ok": False, "message": "合约未配置/地址或ABI有问题"}

    dataset_b32 = Web3.to_bytes(hexstr="0x" + dataset_hash_hex)
    result_b32  = Web3.to_bytes(hexstr="0x" + result_hash_hex)

    accounts = getattr(w3.eth, "accounts", [])
    if not accounts:
        return {"ok": False, "message": "节点没有可用账户（w3.eth.accounts 为空）"}

    sender = accounts[0]
    try:
        nonce = w3.eth.get_transaction_count(sender)

        tx = c.functions.commitReport(dataset_b32, result_b32).build_transaction({
            "from": sender,
            "nonce": nonce,
            "gas": 300000,
            "gasPrice": w3.to_wei(1, "gwei")
        })

        tx_hash = w3.eth.send_transaction(tx)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        return {
            "ok": True,
            "from": sender,
            "tx_hash": tx_hash.hex(),
            "block_number": int(receipt.blockNumber)
        }
    except Exception as e:
        return {"ok": False, "message": f"commitReport 发送失败：{e}"}

