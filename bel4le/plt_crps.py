import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

file_path = 'CRPS_score.xlsx'  # 确保Excel文件在与脚本相同的目录下
df = pd.read_excel(file_path, header=[1])

crps_mean_cca = df.iloc[0, df.columns.str.contains('CCA')].values.astype(float)
crps_mean_mdn = df.iloc[0, df.columns.str.contains('MDN')].values.astype(float)
crps_std_cca = df.iloc[1, df.columns.str.contains('CCA')].values.astype(float)
crps_std_mdn = df.iloc[1, df.columns.str.contains('MDN')].values.astype(float)

rmse_cca = df.iloc[2, df.columns.str.contains('CCA')].values.astype(float)
rmse_mdn = df.iloc[2, df.columns.str.contains('MDN')].values.astype(float)
rmspe_cca = df.iloc[3, df.columns.str.contains('CCA')].values.astype(float)
rmspe_mdn = df.iloc[3, df.columns.str.contains('MDN')].values.astype(float)

groups = {
    'S1': slice(0, 5),
    'S2': slice(5, 10),
    'S3': slice(10, 15)
}

for group, sl in groups.items():
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(8, 6))

    rects1 = ax.bar(np.arange(5) - 0.35/2, rmse_cca[sl], 0.35, color='#4E79A7', edgecolor='black', linewidth=1, alpha=0.85, label='CCA')
    rects2 = ax.bar(np.arange(5) + 0.35/2, rmse_mdn[sl], 0.35, color='#F28E2B', edgecolor='black', linewidth=1, alpha=0.85, label='MDN')

    ax.set_ylim(0, max(max(rmse_cca[sl]), max(rmse_mdn[sl])) * 1.15)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f'{group}-{i+1}' for i in range(5)])

    ax.set_ylabel('RMSE Score')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.legend()

    fig.tight_layout()
    plt.show()

# %%
for group, sl in groups.items():
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(8, 6))

    rects1 = ax.bar(np.arange(5) - 0.35/2, rmspe_cca[sl], 0.35, color='#4E79A7', edgecolor='black', linewidth=1, alpha=0.85, label='CCA')
    rects2 = ax.bar(np.arange(5) + 0.35/2, rmspe_mdn[sl], 0.35, color='#F28E2B', edgecolor='black', linewidth=1, alpha=0.85, label='MDN')

    ax.set_ylim(0, max(max(rmspe_cca[sl]), max(rmspe_mdn[sl])) * 1.15)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f'{group}-{i+1}' for i in range(5)])
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.set_ylabel('RMSPE Score')
    # ax.set_xlabel(f'{group} Scenarios')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.legend()

    fig.tight_layout()
    plt.show()

# %%
for group, sl in groups.items():
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(8, 6))

    rects1 = ax.bar(np.arange(5) - 0.35/2, crps_mean_cca[sl], 0.35, yerr=crps_std_cca[sl], capsize=5,
                    color='#4E79A7', edgecolor='black', linewidth=1, alpha=0.85, label='CCA')
    rects2 = ax.bar(np.arange(5) + 0.35/2, crps_mean_mdn[sl], 0.35, yerr=crps_std_mdn[sl], capsize=5,
                    color='#F28E2B', edgecolor='black', linewidth=1, alpha=0.85, label='MDN')

    # 动态设置y轴范围
    ax.set_ylim(0, max(max(crps_mean_cca[sl] + crps_std_cca[sl]), max(crps_mean_mdn[sl] + crps_std_mdn[sl])) * 1.15)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f'{group}-{i+1}' for i in range(5)])

    ax.set_ylabel('CRPS Score')

    # 添加网格线
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 添加图例
    ax.legend()
    # 调整图表布局
    fig.tight_layout()
    plt.show()