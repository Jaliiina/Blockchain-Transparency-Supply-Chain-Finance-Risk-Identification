from flask import Flask, render_template, jsonify, request
import pandas as pd
from pathlib import Path
from compute import build_factors, top_metrics, high_risk_table
from security import (
    canonical_dumps, sha256_hex, sign_hex, verify, public_key_hex
)

app = Flask(__name__)

DATA_PATH = Path(__file__).parent / "data" / "trust_chain_dataset_2020plus.csv"

@app.route("/")
def index():
    df = pd.read_csv(DATA_PATH)
    df_calc = build_factors(df)
    metrics = top_metrics(df_calc)
    dims = {k: round(float(df_calc[k].mean()),3) for k in ["T1","T2","T3","T4"]}
    return render_template("index.html", metrics=metrics, dims=dims)

@app.route("/api/summary")
def api_summary():
    df = pd.read_csv(DATA_PATH)
    df_calc = build_factors(df)
    return jsonify(top_metrics(df_calc))

@app.route("/api/scatter")
def api_scatter():
    df = pd.read_csv(DATA_PATH)
    df_calc = build_factors(df)
    d = df_calc.sample(n=min(1200, len(df_calc)), random_state=7)[["T_total","R"]]
    return jsonify({"T_total": d["T_total"].tolist(), "R": d["R"].tolist()})

@app.route("/api/highrisk")
def api_highrisk():
    topk = int(request.args.get("topk", 20))
    df = pd.read_csv(DATA_PATH)
    df_calc = build_factors(df)
    tbl = high_risk_table(df_calc, topk=topk)
    return tbl.to_json(orient="records", force_ascii=False)


@app.route("/api/sampletx")
def api_sampletx():
    df = pd.read_csv(DATA_PATH)
    candidates = df[df["Transaction Hash"].astype(str).str.startswith("0x", na=False)]
    row = (candidates.iloc[0] if len(candidates) else df.iloc[0]).to_dict()
    keep = [
        "Transaction ID","Supplier ID","Customer ID","Order Amount",
        "Quantity Shipped","Order Status","Payment Status",
        "Time to Delivery","Quantity Mismatch","Compliance Check",
        "Transaction Hash","Smart Contract Status","Timestamp","Location"
    ]
    obj = {k: row.get(k) for k in keep if k in row}
    return jsonify({
        "tx": obj,
        "pubkey_hex": public_key_hex()
    })

@app.route("/api/hashcheck", methods=["POST"])
def api_hashcheck():
    """
    期望 body:
    {
      "original": {...},   # 原始交易JSON
      "modified": {...}    # 修改后的交易JSON
    }
    返回:
    {
      "original": {"canonical": "...", "hash": "...", "sig": "...", "verify": true/false},
      "modified": {"canonical": "...", "hash": "...", "verify_with_original_sig": true/false},
      "equal_hash": true/false
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    orig = data.get("original") or {}
    mod  = data.get("modified") or {}

    c1 = canonical_dumps(orig)
    c2 = canonical_dumps(mod)

    h1 = sha256_hex(c1)
    h2 = sha256_hex(c2)

    sig = sign_hex(c1)
    v1 = verify(c1, sig)
    v2 = verify(c2, sig)

    return jsonify({
        "original": {"canonical": c1, "hash": h1, "sig": sig, "verify": v1},
        "modified": {"canonical": c2, "hash": h2, "verify_with_original_sig": v2},
        "equal_hash": (h1 == h2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

