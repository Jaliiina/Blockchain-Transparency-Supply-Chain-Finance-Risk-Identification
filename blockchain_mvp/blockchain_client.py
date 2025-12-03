from web3 import Web3
from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware 
from pathlib import Path
import json

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# 告诉 web3：这是 PoA 链（比如 geth --dev / clique）
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)


def chain_info():
    """
    返回链状态信息：是否连接成功、chain_id、区块高度、最新区块哈希
    """
    ok = w3.is_connected()
    if not ok:
        return {"connected": False}

    latest_block = w3.eth.get_block("latest")

    return {
        "connected": True,
        "chain_id": int(w3.eth.chain_id),
        "block_number": int(latest_block.number),
        "latest_block_hash": latest_block.hash.hex(),
    }

CONTRACTS_DIR = Path(__file__).parent / "contracts"
ABI_PATH = CONTRACTS_DIR / "hello_abi.json"


HELLO_CONTRACT_ADDRESS = "0xe4618C3f72446CF62eC0C74ddF6D1def0CC72626"
_hello_contract = None

if ABI_PATH.exists():
    try:
        abi = json.loads(ABI_PATH.read_text(encoding="utf-8"))
        if HELLO_CONTRACT_ADDRESS:
            _hello_contract = w3.eth.contract(
                address=HELLO_CONTRACT_ADDRESS,
                abi=abi
            )
    except Exception:
        _hello_contract = None


def hello_say():
    """
    调用链上的 Hello 合约的 sayHello() 函数
    合约没配置好时返回 None，让前端提示一下
    """
    if not w3.is_connected() or _hello_contract is None:
        return None
    try:
        return _hello_contract.functions.sayHello().call()
    except Exception:
        return None
