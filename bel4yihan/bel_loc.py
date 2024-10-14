import os
import numpy as np
import pandas as pd
import properscoring as ps
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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
random_indices = np.random.choice(data.shape[0], size=2000, replace=False)

data = data[random_indices, :]
X = data[:, 3:]
y = data[:, :3]
X -= X[:, [0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

bel = BEL(
    mode='kde',
    X_pre_processing=Pipeline([('pca', PCA(n_components=0.99))]),
    Y_pre_processing=Pipeline([('pca', PCA(n_components=3))]),
    regression_model=CCA(n_components=3),
    n_comp_cca=3
)

bel_model, canonical_correlations = bel.fit(X_train_scaled, y_train_scaled)

print("Canonical Correlation Coefficients:", canonical_correlations)

n_posts = 100
y_pred_scaled, x_obs_c, cca_samples = bel.predict(X_obs=X_test_scaled, n_posts=n_posts, mode='kde', return_cca=True)

y_pred = np.zeros((y_pred_scaled.shape[1], y_pred_scaled.shape[0], y_pred_scaled.shape[2]))
for i in range(y_pred_scaled.shape[0]):
    original_scale_preds = scaler_y.inverse_transform(y_pred_scaled[i])
    y_pred[:, i, :] = original_scale_preds

# %%
random_index = np.random.randint(0, y_test.shape[0])

plt.figure(figsize=(8, 6))

sns.kdeplot(y_train[:, 0], color='blue', label='Prior', fill=True)
sns.kdeplot(y_pred[:, random_index, 0], color='cyan', label='Posterior', fill=True)

truth_value = y_test[random_index, 0]
plt.axvline(truth_value, color='red', linestyle='-', linewidth=2, label='Test')

plt.legend()
plt.show()

# %%
random_index = np.random.randint(0, y_test.shape[0])

plt.figure(figsize=(8, 6))

# plt.scatter(x=y_train[:, 0], y=y_train[:, 1], color='blue')

sns.kdeplot(x=y_pred[:, random_index, 0], y=y_pred[:, random_index, 2], color='cyan', label='Posterior', fill=True)

truth_value_x = y_test[random_index, 0]
truth_value_y = y_test[random_index, 2]
plt.scatter(truth_value_x, truth_value_y, color='red', label='Test', s=100)
plt.xlim(-5, 155)
plt.ylim(-5, 155)
plt.show()

# %%
random_index = np.random.randint(0, y_test.shape[0])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(y_pred[:, random_index, 0], y_pred[:, random_index, 1], y_pred[:, random_index, 2],
           color='cyan', label='Posterior', alpha=0.2)

truth_value_x = y_test[random_index, 0]
truth_value_y = y_test[random_index, 1]
truth_value_z = y_test[random_index, 2]
ax.scatter(truth_value_x, truth_value_y, truth_value_z, color='red', label='Test', s=100)

ax.set_xlabel('First column')
ax.set_ylabel('Second column')
ax.set_zlabel('Third column')

ax.set_xlim(-5, 155)
ax.set_ylim(-5, 155)
ax.set_zlim(-5, 155)

ax.legend()
plt.show()