import pandas as pd
import numpy as np

def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)

def mean_gap_days(ts: pd.Series):
    ts = pd.to_datetime(ts, errors="coerce").sort_values()
    diffs = ts.diff().dt.days.dropna()
    return diffs.mean() if len(diffs) else np.nan

def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # === R（综合风险，等权六项）===
    risk_payment = df["Payment Status"].astype(str).str.lower().map({"paid": 0.0, "overdue": 1.0, "unpaid": 1.0})
    risk_order = df["Order Status"].astype(str).str.lower().map({"delivered": 0.0, "pending": 0.7, "cancelled": 1.0})
    risk_components = pd.DataFrame({
        "fraud": pd.to_numeric(df.get("Fraud Indicator"), errors="coerce").fillna(0),
        "pay": risk_payment.fillna(0),
        "order": risk_order.fillna(0),
        "qty_mismatch": minmax(df.get("Quantity Mismatch")),
        "ttd": minmax(df.get("Time to Delivery")),
        "compliance_fail": 1 - pd.to_numeric(df.get("Compliance Check"), errors="coerce").fillna(0)
    })
    df["R"] = risk_components.mean(axis=1, skipna=True)

    # === T2（信息完整性）===
    qty_ok = (pd.to_numeric(df.get("Quantity Mismatch"), errors="coerce").fillna(0) == 0).astype(float)
    comp_ok = pd.to_numeric(df.get("Compliance Check"), errors="coerce").fillna(0).clip(0,1).astype(float)
    temp_ok = pd.to_numeric(df.get("Temperature"), errors="coerce").between(-30,60).astype(float)
    hum_ok  = pd.to_numeric(df.get("Humidity"), errors="coerce").between(0,100).astype(float)
    df["T2"] = pd.concat([qty_ok, comp_ok, temp_ok, hum_ok], axis=1).mean(axis=1)

    # === T3（披露频率/节奏）===
    g_sup = df.groupby("Supplier ID")["Timestamp"]
    span_sup = (g_sup.transform("max") - g_sup.transform("min")).dt.days.replace(0, 1)
    cnt_sup  = df.groupby("Supplier ID")["Transaction ID"].transform("count")
    sup_freq = cnt_sup / span_sup
    avg_gap  = g_sup.transform(mean_gap_days)

    T3_s = 0.4*minmax(sup_freq) + 0.4*comp_ok + 0.2*(1 - minmax(avg_gap))
    std = T3_s.std(ddof=0)
    df["T3"] =T3_s

    # === T4（审计可视度）===
    hash_ok = df.get("Transaction Hash").astype(str).str.startswith("0x", na=False).astype(float)
    sc_status = df.get("Smart Contract Status").astype(str).str.lower().map({
        "completed": 1.0, "triggered": 0.8, "active": 0.6
    }).fillna(0.5)
    comp_vis = pd.to_numeric(df.get("Compliance Check"), errors="coerce").fillna(0).clip(0,1).astype(float)
    df["T4"] = pd.concat([sc_status, hash_ok, comp_vis], axis=1).mean(axis=1)

    # === T1（数据可追溯性）===
    gps_present = df.get("GPS Coordinates").astype(str).str.contains(r"\d", regex=True, na=False).astype(float)
    gps_miss = 1 - gps_present
    hash_miss = 1 - hash_ok
    long_ttd  = minmax(df.get("Time to Delivery"))
    qty_mis   = (pd.to_numeric(df.get("Quantity Mismatch"), errors="coerce").fillna(0) > 0).astype(float)
    pending_cancel = df.get("Order Status").astype(str).str.lower().map({"pending":1.0,"cancelled":1.0}).fillna(0.0)

    trace_gap_flags = pd.concat([gps_miss, hash_miss, long_ttd, qty_mis, pending_cancel], axis=1).mean(axis=1)
    T1_v7 = 1 - trace_gap_flags  # 值越大=越可追溯
    std1 = T1_v7.std(ddof=0)
    df["T1"] = T1_v7

    # 总透明度
    df["T_total"] = df[["T1","T2","T3","T4"]].mean(axis=1)
    return df

def top_metrics(df_calc: pd.DataFrame):
    return {
        "n": int(len(df_calc)),
        "T_total_mean": float(df_calc["T_total"].mean()),
        "R_mean": float(df_calc["R"].mean()),
        "q75_R": float(df_calc["R"].quantile(0.75))
    }

def high_risk_table(df_calc: pd.DataFrame, topk=20):
    q75 = df_calc["R"].quantile(0.75)
    cols = [c for c in ["Transaction ID","Supplier ID","Customer ID","R","T_total","Order Status","Payment Status","Transaction Hash"] if c in df_calc.columns]
    return df_calc[df_calc["R"] >= q75][cols].sort_values("R", ascending=False).head(topk)
