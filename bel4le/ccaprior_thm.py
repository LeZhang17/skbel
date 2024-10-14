import numpy as np
import pandas as pd
import properscoring as ps
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from scipy.stats import gaussian_kde
from skbel import BEL

loaded_data = np.loadtxt('A_parameters.txt')
parameters = loaded_data
# ii = parameters[:,7]>0.275
parameters = parameters[:2000, :15]
parameters = np.delete(parameters, 7, axis=1)
y = np.random.randn(2000)
p1, p2, n1, n2 = train_test_split(parameters, y, test_size=0.05, random_state=42)

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

np.random.seed(42)

data = pd.read_pickle('datagt.pkl')
data2 = pd.read_pickle('datags.pkl')

time_steps = 50

x = data[:, :time_steps]
y = data2[:, time_steps - 20:time_steps + 50]
x_train, x_test, y_train_ordi, y_test_ordi = train_test_split(x, y, test_size=0.05, random_state=42)

y_train = y_train_ordi[:, 20:]
y_test = y_test_ordi[:, 20:]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

pca_x = PCA(n_components=6)
pca_y = PCA(n_components=2)

x_train_pca = pca_x.fit_transform(x_train_scaled)
x_test_pca = pca_x.transform(x_test_scaled)
y_train_pca = pca_y.fit_transform(y_train_scaled)

explained_variance_x = pca_x.explained_variance_
weights_x = 1 / np.sqrt(explained_variance_x)  # 以方差的平方根为权重的倒数

def weighted_euclidean_distances(X_obs_pca, x_train_pca, weights):
    delta = (X_obs_pca - x_train_pca) * weights
    return np.sqrt(np.sum(delta**2, axis=1))

closest_indices_list = []
for i in range(x_test_pca.shape[0]):
    X_obs_pca = x_test_pca[i].reshape(1, -1)
    distances = weighted_euclidean_distances(X_obs_pca, x_train_pca, weights_x)
    closest_indices = np.argsort(distances)[:400]
    closest_indices_list.append(closest_indices)
ind=np.array(closest_indices_list)

#%%
y_pred_scaled = []
cca_samples_all = []
canonical_correlations_list = []
X_train_cca_list = []
y_train_cca_list = []
x_test_cca_list = []
y_test_cca_list = []

for i, closest_indices in enumerate(closest_indices_list):
    x_train_closest = x_train_scaled[closest_indices]
    y_train_closest = y_train_scaled[closest_indices]

    bel_closest = BEL(
        mode='kde',
        X_pre_processing=Pipeline([('pca', PCA(n_components=6))]),  # 手动定义的 PCA 模型
        Y_pre_processing=Pipeline([('pca', PCA(n_components=2))]),
        regression_model=CCA(n_components=2),
        n_comp_cca=2
    )

    bel_model_closest, canonical_correlations_closest = bel_closest.fit(x_train_closest, y_train_closest)

    canonical_correlations_list.append(canonical_correlations_closest)
    X_train_cca_list.append(bel_closest.X_f)
    y_train_cca_list.append(bel_closest.Y_f)

    xtest_pca = bel_closest.X_pre_processing.transform(x_test_scaled[i].reshape(1, -1))
    ytest_pca = bel_closest.Y_pre_processing.transform(y_test_scaled[i].reshape(1, -1))

    x_test_cca, y_test_cca = bel_closest.regression_model.transform(xtest_pca, ytest_pca)
    x_test_cca_list.append(x_test_cca)
    y_test_cca_list.append(y_test_cca)

    y_pred_scaled_closest, x_obs_c_closest, cca_samples_closest = bel_closest.predict(
        X_obs=x_test_scaled[i].reshape(1, -1), n_posts=100, mode='kde', return_cca=True
    )

    y_pred_scaled.append(y_pred_scaled_closest)
    cca_samples_all.append(cca_samples_closest)

y_pred_scaled = np.vstack(y_pred_scaled)
cca_samples = np.vstack(cca_samples_all)
canonical_correlations = np.vstack(canonical_correlations_list)
X_train_cca = np.array(X_train_cca_list)  # 形状为 (400, 500, 2)
y_train_cca = np.array(y_train_cca_list)  # 形状为 (400, 500, 2)
x_test_cca = np.array(x_test_cca_list)  # 形状为 (400, 2)
y_test_cca = np.array(y_test_cca_list)  # 形状为 (400, 2)

# %%

y_pred = np.zeros((y_pred_scaled.shape[1], y_pred_scaled.shape[0], y_pred_scaled.shape[2]))
for i in range(y_pred_scaled.shape[0]):
    original_scale_preds = scaler_y.inverse_transform(y_pred_scaled[i])
    y_pred[:, i, :] = original_scale_preds

mean, std = calculate_crps(y_test, y_pred.reshape(100, y_test.shape[0], y_test.shape[1]))
mean_preds = np.mean(y_pred, axis=0)
rmse1 = calculate_rmse(y_test, mean_preds)
rmspe1 = calculate_rmspe(y_test, mean_preds)

print('crps_mean:', mean)
print('crps_std:', std)
print('rmse:', rmse1)
print('rmspe:', rmspe1)
print('cor:', np.mean(canonical_correlations,axis=0))

# %%
random_index = np.random.randint(0, x_test.shape[0])
# random_index = 251

y_true_sample_cca = y_test_cca[random_index,0,:]
y_pred_sample_cca = cca_samples[random_index,:, :]

plt.rcParams.update({'font.size': 20})
for i in range(2):
    g = sns.JointGrid(x=X_train_cca[random_index,:, i], y=y_train_cca[random_index,:, i], space=0, height=8)
    g.plot_joint(sns.scatterplot, color='black', alpha=0.5, s=2, label='Training')

    sns.kdeplot(x=X_train_cca[random_index,:, i], y=y_train_cca[random_index,:, i], ax=g.ax_joint, fill=True, cmap="coolwarm", alpha=0.5)

    sns.kdeplot(y_train_cca[random_index,:, i], ax=g.ax_marg_y, vertical=True, fill=True, color='purple', label='Training Density')

    sns.kdeplot(X_train_cca[random_index,:, i], ax=g.ax_marg_x, fill=True, color='blue', label='Training Density')

    g.ax_joint.scatter(x_test_cca[random_index,:, i], y_test_cca[random_index,:, i], color='red', edgecolor='white', label='Test Sample')
    g.ax_joint.axvline(x_test_cca[random_index,:, i], color='royalblue', lw='1')
    g.ax_joint.axhline(y_test_cca[random_index,:, i], color='purple', lw='1')

    sns.kdeplot(y_pred_sample_cca[:, i], ax=g.ax_marg_y, vertical=True, color='orange', linestyle='--', fill=True, label='Posterior Density')

    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles, labels=labels)
    C = "C"
    g.set_axis_labels(f"$d^{{{C}{i+1}}}$", f"$h^{{{C}{i+1}}}$", fontsize=24)

    plt.subplots_adjust(left=0.15, right=1, top=1, bottom=0.1)
    plt.show()

# %%
random_index = np.random.randint(0, x_test_pca.shape[0])
prior_color = '#A9A9A9'
prior_selected_color = '#87CEEB'
posterior_color = '#6A5ACD'
true_value_color = '#FF6347'

plt.figure(figsize=(8, 6))
time = np.arange(0, 5, 0.1)
time2 = np.arange(-1.9, 0.1, 0.1)

mean_prior = np.mean(y_train, axis=0)
std_prior = np.std(y_train, axis=0)
plt.fill_between(time, mean_prior - 1.96*std_prior, mean_prior + 1.96*std_prior, color=prior_color, alpha=0.3, label='Prior (95% CI)')

mean_prior_selected = np.mean(y_train[ind[random_index]], axis=0)
std_prior_selected = np.std(y_train[ind[random_index]], axis=0)
plt.fill_between(time, mean_prior_selected - 1.96*std_prior_selected, mean_prior_selected + 1.96*std_prior_selected, color=prior_selected_color, alpha=0.5, label='Prior Selected (95% CI)')

for i in range(y_pred.shape[0]):
    plt.plot(time, y_pred[i, random_index, :], color=posterior_color, alpha=0.7, label='Posterior' if i == 0 else "", linewidth=1.5)

plt.plot(time, y_test[random_index], color=true_value_color, label='Test', linewidth=2)
plt.plot(time2, y_test[random_index, :20], color=true_value_color, label='Historical Test', linestyle='--',linewidth=2)

plt.grid(linestyle='--', linewidth=0.5, alpha=0.5)

plt.xlabel("Time (years)")
plt.ylabel("T (\u00B0C)")

plt.tight_layout()
plt.legend(fontsize=14)
plt.show()


#%%
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 22})
random_index = np.random.randint(0, x_test_pca.shape[0])

g = sns.JointGrid(x=x_train_pca[:, 0], y=x_train_pca[:, 1], space=0, height=8)

g.ax_joint.axvline(x_test_pca[random_index, 0], color='purple', linewidth=1, alpha=0.7, ls='--')
g.ax_joint.axhline(x_test_pca[random_index, 1], color='purple', linewidth=1, alpha=0.7, ls='--')

g.ax_joint.scatter(x_train_pca[:, 0], x_train_pca[:, 1], color='royalblue', alpha=0.5, s=10, label="Prior")
g.ax_joint.scatter(x_train_pca[ind[random_index]][:, 0], x_train_pca[ind[random_index]][:, 1], color='darkorange', alpha=0.5, s=15, label="Posterior")
g.ax_joint.scatter(x_test_pca[random_index, 0], x_test_pca[random_index, 1], color='red', edgecolor='white', s=100, label="Test")
g.ax_joint.legend(loc='upper left')

sns.kdeplot(x_train_pca[:, 0], ax=g.ax_marg_x, color='royalblue', linestyle='--', label='Prior', fill=True)
sns.kdeplot(x_train_pca[ind[random_index]][:, 0], ax=g.ax_marg_x, color='darkorange', linestyle='--', label='Posterior', fill=True)
g.ax_marg_x.set_yticks([])
g.ax_marg_x.legend(loc='lower right')
plt.setp(g.ax_marg_x.get_xticklabels(), visible=False)

sns.kdeplot(x_train_pca[:, 1], ax=g.ax_marg_y, color='royalblue', linestyle=':', fill=True, vertical=True)
sns.kdeplot(x_train_pca[ind[random_index]][:, 1], ax=g.ax_marg_y, color='darkorange', linestyle=':', fill=True, vertical=True)
g.ax_marg_y.set_xticks([])
plt.setp(g.ax_marg_y.get_yticklabels(), visible=False)


g.ax_joint.tick_params(axis='x', labelsize=16)
g.ax_joint.tick_params(axis='y', labelsize=16)

g.ax_joint.set_xlabel("PC-1")
g.ax_joint.set_ylabel("PC-2")
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)
g.ax_joint.grid(False)

plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.1)

plt.show()


#%%
fig, axes = plt.subplots(2, 7, figsize=(20, 8))  # 2 行 7 列的子图布局


for i in range(14):
    row = i // 7
    col = i % 7
    ax = axes[row, col]  # 获取子图
    ax.hist(p1[:, i], bins=20, color='blue', alpha=0.5)
    ax.hist(p1[ind[random_index]][:, i], bins=10, color='red', alpha=0.5)
    ax.set_title(f'Feature {i+1}')  # 设置每个子图的标题

# 调整子图布局
plt.tight_layout()
plt.show()