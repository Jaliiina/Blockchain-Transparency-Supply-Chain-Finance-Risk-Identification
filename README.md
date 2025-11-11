# Blockchain Transparency · Supply Chain Finance Risk Identification  
区块链驱动的供应链金融风险识别机制研究与最小可行产品（MVP）

---

## 🌐 项目概述
本项目结合信息经济学与信号传递理论，研究区块链技术如何通过提升**信息透明度**来改善供应链金融的**风险识别机制**。  
研究从理论、实证与工程实现三个层面展开：

1. **理论层面**  
   构建 “区块链 — 信息透明度 — 风险识别” 分析框架，提出透明度四维度指标体系（T₁–T₄）。  
2. **实证层面**  
   通过回归分析、机器学习模型与可解释性方法（RF、XGBoost、SHAP、Logit、Spearman）验证透明度对风险的影响。  
3. **实现层面（MVP）**  
   基于 Flask + Plotly 构建轻量级可视化原型系统，实现透明度–风险实时展示与区块链哈希验证演示。

---

## 🧩 项目结构
📦 blockchain_mvp/
│
├── data/
│ └── trust_chain_dataset_2020plus.csv # 供应链交易数据集
│
├── static/
│ └── style.css # 前端样式
│
├── templates/
│ ├── base.html # 基础模板
│ └── index.html # 仪表盘与交互页面
│
├── app.py # Flask 启动入口
├── compute.py # 透明度与风险计算逻辑
├── security.py # 哈希与签名验证模块
│
└── transparancy/ # 实验与论文分析结果
├── trust_chain_dataset.csv # 原始数据
├── ml_compare.py # 模型性能对比脚本
├── s3_f.py # 风险值与透明度计算函数
├── pic_2.py / pic.py # 绘图脚本
├── feature_importance_rf.csv # 随机森林特征重要性
├── feature_importance_xgb.csv # XGBoost 特征重要性
├── feature_importance_xgb_shap.csv # SHAP 可解释性结果
├── logit_coefficients.csv # Logit 回归系数
├── ols_T14_tuned_coefs.csv # OLS 回归结果
├── ml_cv_results.csv # 模型交叉验证结果
└── spearman_corr.csv # 相关系数矩阵

---
## 📊 MVP 功能展示

### 1️⃣ 透明度与风险仪表盘

- 实时展示样本量、平均透明度、平均风险及高风险阈值；
- 绘制交互式 “透明度–风险” 散点图；
- 四维透明度均值：
  - **T1 可追溯性**：0.647  
  - **T2 完整性**：0.647  
  - **T3 披露频率**：0.514  
  - **T4 审计可视度**：0.771  

### 2️⃣ 高风险交易识别表

自动筛选风险值 **R** 位于前 25% 的交易记录并展示关键信息。

### 3️⃣ 哈希与签名验证演示

- 计算交易 JSON 的 **SHA-256 哈希**；
- 生成 **ECDSA 数字签名** 并实时验签；
- 当字段被篡改时，系统立即显示 **“哈希不匹配 / 验签失败”**，可视化演示区块链数据不可篡改性。

---

## 🧠 实验部分（transparancy/）

包含论文实验及统计结果文件：

- **OLS / Logit 回归分析**：验证透明度维度对风险的显著性；
- **随机森林 & XGBoost**：量化不同特征的重要性；
- **SHAP 可解释性分析**：识别透明度对模型预测的影响路径；
- **相关性分析（Spearman）**：探索多维透明度指标的内在关系。

---

## 📘 理论与意义

该研究证明：

> 区块链通过提升信息可追溯性、完整性、披露频率与可验证性，显著降低了供应链金融的信用与交易风险。  

MVP 原型展示了“从信号传递理论到工程落地”的完整路径，为后续智能合约自动授信、跨链风控等应用提供技术基础。

