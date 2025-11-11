import json, hashlib
from ecdsa import SigningKey, NIST256p, BadSignatureError

_SEED = b"demo-seed-for-risk-mvp-only"
_sk = SigningKey.from_string(hashlib.sha256(_SEED).digest(), curve=NIST256p)
_vk = _sk.get_verifying_key()

def canonical_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sign_hex(text: str) -> str:
    sig = _sk.sign(text.encode("utf-8"))
    return sig.hex()

def verify(text: str, sig_hex: str) -> bool:
    try:
        _vk.verify(bytes.fromhex(sig_hex), text.encode("utf-8"))
        return True
    except BadSignatureError:
        return False

def public_key_hex() -> str:
    return _vk.to_string().hex()
