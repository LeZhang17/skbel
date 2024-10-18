import os
import numpy as np
import pandas as pd
import properscoring as ps
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from scipy.stats import gaussian_kde
from skbel import BEL

np.random.seed(42)

file = r"\\record-5.csv"
path = os.getcwd()+file
datasets = pd.read_csv(path, header=0)
data = np.array(datasets)

file_r = r"\\lead_breaking_data.csv"
path = os.getcwd()+file_r
data_r = pd.read_csv(path, header=0)
test = np.array(data_r)
test = np.delete(test, [2,3,4], axis=1)

for group in range(1, 60):
    group_rows = test[test[:, 0] == group]
    sorted_indices = np.argsort(group_rows[:, 2])
    test[test[:, 0] == group] = group_rows[sorted_indices]

test = test[:, 1].reshape(59, 11)
test -= np.min(test, axis=1, keepdims=True)


X = data[:, 3:]
y = data[:, :3]
X -= np.min(X, axis=1, keepdims=True)

i = 10
sns.kdeplot(test[:,i],label='sm')
sns.kdeplot(X[:,i],label='leading break')
plt.title(f"sensor {i+1} Arrival time")
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)
X_true_scaled = scaler_x.transform(test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

yt_pred = []
n_posts = 200
# 针对每一个 X_true_scaled 中的样本
for i in range(2):
    X_true_sample = X_true_scaled[i].reshape(1, -1)  # 当前真实样本

    # 计算该真实样本与训练集中所有样本的欧氏距离
    distances = cdist(X_train_scaled, X_true_sample, metric='euclidean')

    # 挑选出距离最近的2000个训练集样本的索引
    closest_indices = np.argsort(distances.ravel())[:2000]

    # 使用最近的2000个样本进行局部训练
    X_train_local = X_train_scaled[closest_indices]
    y_train_local = y_train_scaled[closest_indices]

    # 创建一个局部的 BEL 模型
    bel_local = BEL(
        mode='kde',
        X_pre_processing=Pipeline([('pca', PCA(n_components=10))]),
        Y_pre_processing=Pipeline([('pca', PCA(n_components=3))]),
        regression_model=CCA(n_components=3),
        n_comp_cca=3
    )

    # 对局部训练集进行拟合
    bel_model_local, canonical_correlations = bel_local.fit(X_train_local, y_train_local)

    # 对当前的真实样本进行预测
    yt_pred_scaled_local, _, _ = bel_local.predict(X_obs=X_true_sample, n_posts=n_posts, mode='kde', return_cca=True)

    # 反标准化预测结果，保留二维结构: (n_posts, n_targets)
    yt_pred_local = scaler_y.inverse_transform(yt_pred_scaled_local[0])

    # 将二维的预测结果保存到总的预测列表中
    yt_pred.append(yt_pred_local)

# 将所有结果转换为三维数组，形状为 (n_samples, n_posts, n_targets)
yt_pred = np.array(yt_pred)

# %%
file_t = r"\\sources.csv"
path = os.getcwd()+file_t
data_t = pd.read_csv(path, header=0)
testx = np.array(data_t)
testx = np.delete(testx, 0, axis=1)

random_index = np.random.randint(0, testx.shape[0])
random_index = 0
fig, axes = plt.subplots(3, 1)

sns.kdeplot(y_train[:, 0], color='blue', label='Prior', fill=True, ax=axes[0])
sns.kdeplot(yt_pred[:, random_index, 0], color='cyan', label='Posterior', fill=True, ax=axes[0])
truth_value = testx[random_index, 0]
axes[0].axvline(truth_value, color='red', linestyle='-', linewidth=2, label='Test')
axes[0].legend()

sns.kdeplot(y_train[:, 1], color='blue', label='Prior', fill=True, ax=axes[1])
sns.kdeplot(yt_pred[:, random_index, 1], color='cyan', label='Posterior', fill=True, ax=axes[1])
truth_value = testx[random_index, 1]
axes[1].axvline(truth_value, color='red', linestyle='-', linewidth=2, label='Test')
axes[1].legend()

sns.kdeplot(y_train[:, 2], color='blue', label='Prior', fill=True, ax=axes[2])
sns.kdeplot(yt_pred[:, random_index, 2], color='cyan', label='Posterior', fill=True, ax=axes[2])
truth_value = testx[random_index, 2]
axes[2].axvline(truth_value, color='red', linestyle='-', linewidth=2, label='Test')
axes[2].legend()


plt.tight_layout()
plt.show()

# %%

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(yt_pred[:, random_index, 0], yt_pred[:, random_index, 1], yt_pred[:, random_index, 2],
           color='cyan', label='Posterior', alpha=0.2)

truth_value_x = y_test[random_index, 0]
truth_value_y = y_test[random_index, 1]
truth_value_z = y_test[random_index, 2]
ax.scatter(truth_value_x, truth_value_y, truth_value_z, color='red', label='Test', s=100)

ax.set_xlabel('First column')
ax.set_ylabel('Second column')
ax.set_zlabel('Third column')

ax.set_xlim(-15, 165)
ax.set_ylim(-15, 165)
ax.set_zlim(-15, 165)

ax.legend()
plt.show()

