import pandas as pd, numpy as np, statsmodels.api as sm
import scipy.stats as st

CSV_PATH = "trust_chain_dataset_2020plus.csv"
OUT_PATH = "ols_T14_tuned_coefs.csv"

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

def main():
    df = pd.read_csv(CSV_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # === R（风险）===
    risk_payment = df["Payment Status"].astype(str).str.lower().map({"paid": 0.0, "overdue": 1.0, "unpaid": 1.0})
    risk_order = df["Order Status"].astype(str).str.lower().map({"delivered": 0.0, "pending": 0.7, "cancelled": 1.0})
    risk_components = pd.DataFrame({
        "fraud": pd.to_numeric(df["Fraud Indicator"], errors="coerce").fillna(0),
        "pay": risk_payment.fillna(0),
        "order": risk_order.fillna(0),
        "qty_mismatch": minmax(df["Quantity Mismatch"]),
        "ttd": minmax(df["Time to Delivery"]),
        "compliance_fail": 1 - pd.to_numeric(df["Compliance Check"], errors="coerce").fillna(0)
    })
    df["R"] = risk_components.mean(axis=1, skipna=True)

    # === T2（信息完整性）===
    qty_ok = (pd.to_numeric(df["Quantity Mismatch"], errors="coerce").fillna(0) == 0).astype(float)
    comp_ok = pd.to_numeric(df["Compliance Check"], errors="coerce").fillna(0).clip(0,1).astype(float)
    temp_ok = pd.to_numeric(df["Temperature"], errors="coerce").between(-30,60).astype(float)
    hum_ok  = pd.to_numeric(df["Humidity"], errors="coerce").between(0,100).astype(float)
    df["T2"] = pd.concat([qty_ok, comp_ok, temp_ok, hum_ok], axis=1).mean(axis=1)

    # === T3（披露频率/节奏，tuned）===
    g_sup = df.groupby("Supplier ID")["Timestamp"]
    span_sup = (g_sup.transform("max") - g_sup.transform("min")).dt.days.replace(0, 1)
    cnt_sup  = df.groupby("Supplier ID")["Transaction ID"].transform("count")
    sup_freq = cnt_sup / span_sup
    avg_gap  = g_sup.transform(mean_gap_days)

    T3_s = 0.4*minmax(sup_freq) + 0.4*comp_ok + 0.2*(1 - minmax(avg_gap))
    df["T3"] = (T3_s - T3_s.mean()) / T3_s.std(ddof=0)

    # === T4（审计可视度）===
    hash_ok = df["Transaction Hash"].astype(str).str.startswith("0x", na=False).astype(float)
    sc_status = df["Smart Contract Status"].astype(str).str.lower().map({
        "completed": 1.0, "triggered": 0.8, "active": 0.6
    }).fillna(0.5)
    comp_vis = pd.to_numeric(df["Compliance Check"], errors="coerce").fillna(0).clip(0,1).astype(float)
    df["T4"] = pd.concat([sc_status, hash_ok, comp_vis], axis=1).mean(axis=1)

    # === T1（数据可追溯性，v7）===
    gps_present = df["GPS Coordinates"].astype(str).str.contains(r"\d", regex=True, na=False).astype(float)
    gps_miss = 1 - gps_present
    hash_miss = 1 - hash_ok
    long_ttd  = minmax(df["Time to Delivery"])
    qty_mis   = (pd.to_numeric(df["Quantity Mismatch"], errors="coerce").fillna(0) > 0).astype(float)
    pending_cancel = df["Order Status"].astype(str).str.lower().map({"pending":1.0,"cancelled":1.0}).fillna(0.0)

    trace_gap_flags = pd.concat([gps_miss, hash_miss, long_ttd, qty_mis, pending_cancel], axis=1).mean(axis=1)
    T1_v7 = 1 - trace_gap_flags 
    df["T1"] = (T1_v7 - T1_v7.mean()) / T1_v7.std(ddof=0)

    # === OLS: R ~ T1+T2+T3+T4（HC3稳健）===
    work = df[["R","T1","T2","T3","T4"]].replace([np.inf,-np.inf], np.nan).dropna()
    X = (work[["T1","T2","T3","T4"]] - work[["T1","T2","T3","T4"]].mean()) / work[["T1","T2","T3","T4"]].std(ddof=0).replace(0,1)
    X2 = sm.add_constant(X)
    y = work["R"].astype(float)

    ols = sm.OLS(y, X2).fit().get_robustcov_results(cov_type="HC3")
    out = pd.DataFrame({"term": X2.columns, "coef_std": ols.params, "pval_HC3": ols.pvalues})
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(out)

    # 总透明度指数
    df["T_total"] = df[["T1","T2","T3","T4"]].mean(axis=1)

    # 去缺失
    data = df[["T_total","R"]].replace([np.inf,-np.inf], np.nan).dropna()

    # === Spearman 等级相关 ===
    rho, pval = st.spearmanr(data["T_total"], data["R"])

    print(f"Spearman ρ = {rho:.4f}")
    print(f"p-value = {pval:.2e}")

if __name__ == "__main__":
    main()

'''
    term  coef_std      pval_HC3
0  const  0.453771  0.000000e+00
1     T1 -0.072532  1.687675e-92
2     T2 -0.019958  1.022999e-03
3     T3 -0.013002  1.976231e-03
4     T4 -0.057270  3.426923e-20
Spearman ρ = -0.5891
p-value = 1.71e-174
'''