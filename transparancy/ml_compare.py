import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.base import clone

import xgboost as xgb_lib
import shap, matplotlib
import matplotlib.pyplot as plt




CSV_PATH = "trust_chain_dataset_2020plus.csv"
OUT_DIR = "D:\桌面\区块链"
os.makedirs(OUT_DIR, exist_ok=True)

# 构造因子
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
    T1_v7 = 1 - trace_gap_flags  # 值越大=越可追溯
    df["T1"] = (T1_v7 - T1_v7.mean()) / T1_v7.std(ddof=0)

    # 特征矩阵
    work = df[["R","T1","T2","T3","T4"]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    return work

#评估工具
def _rmse_compat(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def regression_cv_scores(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": _rmse_compat(y_true, y_pred),
    }

def run_regression_cv(model, X, y, n_splits=10, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = {"R2": [], "MAE": [], "RMSE": []}
    for tr, te in kf.split(X):
        m = clone(model)
        m.fit(X[tr], y[tr])
        yhat = m.predict(X[te])
        sc = regression_cv_scores(y[te], yhat)
        for k in metrics:
            metrics[k].append(sc[k])
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}

def run_logit_cv(X, y_bin, n_splits=10, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = StandardScaler()
    aucs, f1s, pres, recs = [], [], [], []
    coefs = []

    for tr, te in skf.split(X, y_bin):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_bin[tr], y_bin[te]

        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(X_trs, y_tr)
        prob = clf.predict_proba(X_tes)[:,1]
        pred = (prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_te, prob))
        f1s.append(f1_score(y_te, pred, zero_division=0))
        pres.append(precision_score(y_te, pred, zero_division=0))
        recs.append(recall_score(y_te, pred, zero_division=0))
        coefs.append(clf.coef_.reshape(-1))

    scores = {
        "AUC": (np.mean(aucs), np.std(aucs)),
        "F1": (np.mean(f1s), np.std(f1s)),
        "Precision": (np.mean(pres), np.std(pres)),
        "Recall": (np.mean(recs), np.std(recs)),
    }
    coef_mean = np.mean(np.vstack(coefs), axis=0)
    coef_std = np.std(np.vstack(coefs), axis=0)
    return scores, coef_mean, coef_std

def save_importance_csv(names, importances, path):
    pd.DataFrame({"feature": names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(path, index=False, encoding="utf-8-sig")

# -------- 主流程 --------
def main():
    df = pd.read_csv(CSV_PATH)
    work = build_factors(df)

    # 特征与目标
    X_df = work[["T1","T2","T3","T4"]]
    y_reg = work["R"].values
    X = X_df.values

    # ========== 回归：RF 与 XGB ==========
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    rf_scores = run_regression_cv(rf, X, y_reg, n_splits=10, random_state=42)
    xgb_scores = run_regression_cv(xgb, X, y_reg, n_splits=10, random_state=42)

    rf.fit(X, y_reg)
    xgb.fit(X, y_reg)
       
    matplotlib.use("Agg")
    plt.rcParams["font.sans-serif"] = ["SimHei"]     
    plt.rcParams["axes.unicode_minus"] = False        
    max_n = 2000
    if X.shape[0] > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X_vis = X[idx]
        Xdf_vis = X_df.iloc[idx].reset_index(drop=True)
    else:
        X_vis = X
        Xdf_vis = X_df.reset_index(drop=True)

    # 计算 SHAP 值
    booster = xgb.get_booster()
    dmat = xgb_lib.DMatrix(X_vis, feature_names=list(Xdf_vis.columns))
    shap_matrix = booster.predict(dmat, pred_contribs=True)
    shap_values = shap_matrix[:, :-1] 
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        Xdf_vis,
        show=False,
        plot_size=(8, 5),
        color_bar=True
    )
    ax = plt.gca()  
    ax.set_xlabel("SHAP值（特征影响程度）")  
    ax.set_ylabel("特征名称") 
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "xgb_shap_summary_beeswarm.png"), dpi=300)
    plt.close()

    print("已生成：", os.path.join(OUT_DIR, "xgb_shap_summary_beeswarm.png"))

    save_importance_csv(X_df.columns, rf.feature_importances_, os.path.join(OUT_DIR, "feature_importance_rf.csv"))
    try:
        save_importance_csv(X_df.columns, xgb.feature_importances_, os.path.join(OUT_DIR, "feature_importance_xgb.csv"))
    except Exception:
        pass


    # ========== 分类：Logistic（高风险识别） ==========
    HIGH_RISK_QUANTILE = 0.75
    thresh = np.quantile(y_reg, HIGH_RISK_QUANTILE)
    y_bin = (y_reg >= thresh).astype(int)

    logit_scores, logit_coef_mean, logit_coef_std = run_logit_cv(X, y_bin, n_splits=10, random_state=42)

    rows = []

    def add_rows(name, d):
        for k, (m, s) in d.items():
            rows.append({"model": name, "metric": k, "mean": m, "std": s})

    add_rows("RandomForestRegressor", rf_scores)
    add_rows("XGBoostRegressor", xgb_scores)
    for k, (m, s) in logit_scores.items():
        rows.append({"model": "LogisticRegression(q>=%.2f)" % HIGH_RISK_QUANTILE, "metric": k, "mean": m, "std": s})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT_DIR, "ml_cv_results.csv"), index=False, encoding="utf-8-sig")

    # Logit 系数表
    coef_df = pd.DataFrame({
        "feature": ["T1","T2","T3","T4"],
        "coef_mean": logit_coef_mean,
        "coef_std": logit_coef_std
    }).sort_values("coef_mean", ascending=False)
    coef_df.to_csv(os.path.join(OUT_DIR, "logit_coefficients.csv"), index=False, encoding="utf-8-sig")

    print("=== Cross-Validated Results (10-fold) ===")
    print(res_df.pivot_table(index="model", columns="metric", values="mean"))
    print("\n阈值(高风险) R-quantile = %.2f, 数量: %d / %d" % (HIGH_RISK_QUANTILE, y_bin.sum(), len(y_bin)))
    print("\nLogit 标准化系数（均值±std）:")
    print(coef_df)

if __name__ == "__main__":
    main()
