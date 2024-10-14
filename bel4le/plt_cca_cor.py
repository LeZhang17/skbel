import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


one_d = [0.860854, 0.84447526, 0.82881308, 0.81758463, 0.80746653]
two_d = [0.33393601, 0.35862225, 0.41354256, 0.46714032, 0.4539]

one_d = [0.99020505, 0.99480617, 0.99734316, 0.99832088, 0.99852751]
two_d = [0.37467694, 0.4015778, 0.5045988, 0.58124385, 0.59904043]
#
one_d = [0.99993903, 0.99996675, 0.99996883, 0.99997641, 0.99998189]
two_d = [0.93445671, 0.89089316, 0.87282136, 0.88227584, 0.90575834]

labels = [f'S3-{i+1}' for i in range(len(one_d))]
x = np.arange(len(labels))  # x轴位置
width = 0.35  # 柱状图宽度

plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(figsize=(8, 6))

# 渐变颜色的生成
def gradient_color(c1, c2, n):
    return [c1 * (1 - i / (n - 1)) + c2 * i / (n - 1) for i in range(n)]

# 使用渐变配色

colors_one_d = gradient_color(np.array([0.31, 0.53, 0.74]), np.array([0.25, 0.47, 0.71]), len(one_d))
colors_two_d = gradient_color(np.array([0.95, 0.58, 0.24]), np.array([0.93, 0.55, 0.22]), len(two_d))

rects1 = ax.bar(x - width/2, one_d, width, color=colors_one_d, edgecolor='black', linewidth=1, alpha=0.85, label='1D')
rects2 = ax.bar(x + width/2, two_d, width, color=colors_two_d, edgecolor='black', linewidth=1, alpha=0.85, label='2D')

# 动态设置y轴范围
ax.set_ylim(0, 1.15)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.yaxis.set_ticklabels([])
ax.set_ylabel('CCA Correlation Coefficient')

ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.legend(loc='lower right')

# 为数据标签添加背景
for rects, colors in zip([rects1, rects2], [colors_one_d, colors_two_d]):
    for rect, color in zip(rects, colors):
        height = rect.get_height()
        # bbox_props = dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor='white', lw=0.5)
        ax.annotate(f'{height*100:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(10, 5),  # 调整偏移量
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=15, color='black')

# 去除顶部和右侧边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# 调整图表布局
fig.tight_layout()

# 显示图像
plt.show()
