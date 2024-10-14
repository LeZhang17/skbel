import numpy as np
import pandas as pd
import properscoring as ps
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from scipy.stats import gaussian_kde
from skbel import BEL

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def calculate_crps(y_true, y_preds):
    n_obs, n_features = y_true.shape
    crps_values = np.zeros(n_features)

    for i in range(n_features):
        for j in range(n_obs):
            crps_values[i] += ps.crps_ensemble(y_true[j, i], y_preds[:, j, i])
        crps_values[i] /= n_obs

    crps_mean = np.mean(crps_values)
    crps_std = np.std(crps_values)

    return crps_mean, crps_std

# 生成示例数据
np.random.seed(42)

data = pd.read_pickle('datagt.pkl')
data2 = pd.read_pickle('datags.pkl')

time_steps = 150

x = data[:, :time_steps]
y = data[:, time_steps:time_steps + 50]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 初始化BEL对象
bel = BEL(
    mode='kde',  # 可以选择'mvn'或'kde'
    X_pre_processing=Pipeline([('pca', PCA(n_components=6))]),
    Y_pre_processing=Pipeline([('pca', PCA(n_components=2))]),
    regression_model=CCA(n_components=2),
    n_comp_cca=2
)

bel_model, canonical_correlations = bel.fit(x_train_scaled, y_train_scaled)

print("Canonical Correlation Coefficients:", canonical_correlations)

# %%
n_posts = 100

y_pred_scaled, x_obs_c, cca_samples = bel.predict(X_obs=x_test_scaled, n_posts=n_posts, mode='kde', return_cca=True)
y_pred = np.zeros((y_pred_scaled.shape[1], y_pred_scaled.shape[0], y_pred_scaled.shape[2]))

for i in range(y_pred_scaled.shape[0]):
    original_scale_preds = scaler_y.inverse_transform(y_pred_scaled[i])
    y_pred[:, i, :] = original_scale_preds

mean, std = calculate_crps(y_test, y_pred.reshape(n_posts, y_test.shape[0], y_test.shape[1]))

mean_preds = np.mean(y_pred, axis=0)
rmse1 = calculate_rmse(y_test, mean_preds)
rmspe1 = calculate_rmspe(y_test, mean_preds)

print('crps_mean:', mean)
print('crps_std:', std)

print('rmse:', rmse1)
print('rmspe:', rmspe1)
# %%
n_samples, n_posts, n_components = cca_samples.shape
cca_samples_reshaped = cca_samples.reshape(-1, n_components)

_, pca_samples_reshaped = bel.regression_model.inverse_transform(
    np.zeros((cca_samples_reshaped.shape[0], bel.regression_model.n_components)),
    cca_samples_reshaped
)

# 将结果重塑回 3D 数组 (n_samples, n_posts, pca_components)
y_pred_pca = pca_samples_reshaped.reshape(n_samples, n_posts, -1)

x_train_pca = bel.X_pre_processing.transform(x_train_scaled)
y_train_pca = bel.Y_pre_processing.transform(y_train_scaled)
x_test_pca = bel.X_pre_processing.transform(x_test_scaled)
y_test_pca = bel.Y_pre_processing.transform(y_test_scaled)

# 获取CCA空间内的训练集和测试集数据
x_train_cca, y_train_cca = bel.regression_model.transform(x_train_pca, y_train_pca)
x_test_cca, y_test_cca = bel.regression_model.transform(x_test_pca, y_test_pca)


# %%绘制结果# 随机选择一个测试样本索引
# random_index = np.random.randint(0, x_test.shape[0])
random_index = 203
y_true_sample_cca = y_test_cca[random_index]
y_pred_sample_cca = cca_samples[random_index,:, :]

plt.rcParams.update({'font.size': 22})
for i in range(2):
    g = sns.JointGrid(x=x_train_cca[:, i], y=y_train_cca[:, i], space=0, height=8)

    g.plot_joint(sns.scatterplot, color='black', alpha=0.5, s=2, label='Prior')

    sns.kdeplot(x=x_train_cca[:, i], y=y_train_cca[:, i], ax=g.ax_joint, fill=True, cmap="coolwarm", alpha=0.5)

    sns.kdeplot(y_train_cca[:, i], ax=g.ax_marg_y, vertical=True, fill=True, color='royalblue', linestyle='--', label='Prior')
    sns.kdeplot(x_train_cca[:, i], ax=g.ax_marg_x, fill=True, color='royalblue', linestyle='--', label='Prior')


    g.ax_joint.axvline(x_test_cca[random_index, i], color='purple', linewidth=1.5, alpha=0.7, label='Observation')
    g.ax_joint.axhline(y_test_cca[random_index, i], color='purple', linewidth=1, alpha=0.7, ls='--')
    g.ax_joint.scatter(x_test_cca[random_index, i], y_test_cca[random_index, i], color='red', edgecolor='white',
                       label='Test')


    sns.kdeplot(y_pred_sample_cca[:, i], ax=g.ax_marg_y, vertical=True, color='darkorange', linestyle='--', fill=True, label='Posterior')
    g.ax_marg_y.legend(loc='lower right')

    g.ax_joint.tick_params(axis='both', which='major', labelsize=16)
    g.ax_marg_x.tick_params(axis='x', which='major', labelsize=16)
    g.ax_marg_y.tick_params(axis='y', which='major', labelsize=16)

    g.ax_joint.set_xlim(-2.8, 2.2)
    g.ax_joint.set_ylim(-2.8, 1.9)
    # g.ax_joint.set_xlim(-2.8, 5.1)
    # g.ax_joint.set_ylim(-2.8, 3.7)
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles, labels=labels,loc='upper left')
    C = "C"
    g.set_axis_labels(f"$d^{{{C}{i+1}}}$", f"$h^{{{C}{i+1}}}$", fontsize=24)
    plt.subplots_adjust(left=0.15, right=1, top=1, bottom=0.1)
    plt.show()

#%%
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 22})
time = np.arange(0.1, 5.1, 0.1)
for i in range(y_train.shape[0]):
    plt.plot(time, y_train[i], color='gray', alpha=0.3, lw=0.7, label='Prior' if i == 0 else "")

for i in range(y_pred.shape[0]):
    plt.plot(time, y_pred[i, random_index, :], color='royalblue', alpha=0.7, label='Posterior' if i == 0 else "")

plt.plot(time, y_test[random_index], color='firebrick', label='Test', linewidth=3)
plt.grid(linestyle='--', linewidth=0.5, alpha=0.5)
plt.xlabel("Time (years)")
plt.ylabel("T (\u00B0C)")
# plt.ylabel(r"$\phi_{mob}$ (°)")
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

# #%% Extract the selected sample's true value, prior and posterior distributions
# y_train_pca_sample = y_train_pca  # Prior: all training samples in PCA space
# y_pred_sample_pca = y_pred_pca[random_index, :, :]
# y_test_sample_pca = y_test_pca[random_index, :]
#
# plt.rcParams.update({'font.size': 22})
# # Create JointGrid
# g = sns.JointGrid(x=y_train_pca_sample[:, 0], y=y_train_pca_sample[:, 1], space=0, height=8)
#
# # Draw lines connecting the true value to the axes
# g.ax_joint.axvline(y_test_sample_pca[0], color='royalblue',  linewidth=1)
# g.ax_joint.axhline(y_test_sample_pca[1], color='purple',  linewidth=1)
#
# # Main scatter plot
# g.ax_joint.scatter(y_train_pca_sample[:, 0], y_train_pca_sample[:, 1], color='royalblue', alpha=0.5, s=10, label="Prior")
# g.ax_joint.scatter(y_pred_sample_pca[:, 0], y_pred_sample_pca[:, 1], color='darkorange', alpha=0.5, s=15, label="Posterior")
# g.ax_joint.scatter(y_test_sample_pca[0], y_test_sample_pca[1], color='red', edgecolor='white', label="True", s=100)
# # g.ax_joint.legend(loc='upper right')
#
# # Density plot on the x-axis
# sns.kdeplot(y_train_pca_sample[:, 0], ax=g.ax_marg_x, color='royalblue', linestyle='--', label='Prior', fill=True)
# sns.kdeplot(y_pred_sample_pca[:, 0], ax=g.ax_marg_x, color='darkorange', linestyle='--', label='Posterior', fill=True)
# g.ax_marg_x.set_yticks([])
# # g.ax_marg_x.legend(loc='lower right')
# plt.setp(g.ax_marg_x.get_xticklabels(), visible=False)
#
# # Density plot on the y-axis
# sns.kdeplot(y_train_pca_sample[:, 1], ax=g.ax_marg_y, color='royalblue', linestyle='--', fill=True, vertical=True)
# sns.kdeplot(y_pred_sample_pca[:, 1], ax=g.ax_marg_y, color='darkorange', linestyle='--', fill=True, vertical=True)
# g.ax_marg_y.set_xticks([])
# plt.setp(g.ax_marg_y.get_yticklabels(), visible=False)
# # g.ax_joint.set_xlim(-2.5, 4.5)
# # g.ax_joint.set_ylim(-2.5, 3.5)
# # Set labels and title
# g.ax_joint.set_xlabel("PC-1")
# g.ax_joint.set_ylabel("PC-2")
#
# plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.1)
#
# plt.show()
