import matplotlib.pyplot as plt

# 设置中文字体及显示参数
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'  # 确保下标显示正常

# 数据准备（变量顺序：从上到下为T₁→T₂→T₃→T₄）


variables = [
    r'T$_4$（审计可视度）' ,
    r'T$_3$（披露频率与节奏）',
    r'T$_2$（信息完整性）',
    r'T$_1$（数据可追溯性）',   # 最上方    # 最下方
]
  # 对应变量的标准化系数
coefficients = [-0.0573, -0.0130, -0.0200, -0.0725]  # 对应变量的标准化系数
# 新配色（按T1→T4顺序对应）
colors = ['#71A682', '#81989B', '#D19246', '#B5AF8B']

# 创建画布，绘制水平条形图（柱子从右往左延伸，匹配负值方向）
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(variables, coefficients, color=colors, height=0.6, edgecolor='black', linewidth=0.5)

# 添加系数值标签（显示在柱子左侧，保留4位小数）
for bar in bars:
    width = bar.get_width()  # 系数值（负值）
    # 标签位置：在柱子末端左侧偏移，避免重叠
    ax.text(width - 0.004, bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            ha='right', va='center', fontsize=10)

# 设置坐标轴与标题
ax.set_title('各变量标准化系数（β）', fontsize=14)
ax.set_xlabel('标准化系数（β）', fontsize=12)
ax.set_xlim(-0.085, 0)  # 调整x轴范围，适配系数大小
ax.grid(axis='x', linestyle='--', alpha=0.7)  # x轴网格线辅助读数

# 优化布局
plt.tight_layout()

# 显示图形
plt.show()