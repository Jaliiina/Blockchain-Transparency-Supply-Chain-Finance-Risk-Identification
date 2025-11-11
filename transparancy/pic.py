import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体及显示参数
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'  # 确保下标显示正常

# 准备数据（T后的数字用TeX语法设置为下标）
features = [r'T$_1$（可追溯性）', r'T$_2$（完整性）', r'T$_3$（披露频率）', r'T$_4$（审计可视度）']
xgb_importance = [0.200, 0.530, 0.066, 0.204]
rf_importance = [0.356, 0.300, 0.176, 0.167]

# 调整配色（深紫与橙色，对比鲜明且视觉舒适）
xgb_color = "#51999F"  # 深紫色
rf_color = "#EA9E58"   # 珊瑚橙

# 条形图参数
x = np.arange(len(features))
width = 0.35

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制条形图
rects1 = ax.bar(x - width/2, xgb_importance, width, label='XGBoost 重要性', color=xgb_color, alpha=0.85)
rects2 = ax.bar(x + width/2, rf_importance, width, label='随机森林 重要性', color=rf_color, alpha=0.85)

# 图表标签设置
ax.set_title('不同模型的特征重要性对比', fontsize=15, pad=15)
ax.set_xlabel('特征', fontsize=12, labelpad=10)
ax.set_ylabel('重要性值', fontsize=12, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=10, ha='right', fontsize=11)  # 轻微旋转标签
ax.legend(fontsize=11)

# 条形上方添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# 调整布局
fig.tight_layout()

# 显示图形
plt.show()