'''
California房价数据集,多特征房价数据集，使用线性回归模型进行预测
'''
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score

# 加载California房价数据集
data = sklearn.datasets.fetch_california_housing()
X = data.data
y = data.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征工程 — use training-set statistics for both splits
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

# ---- Train ----
ridge = Ridge(alpha=0.5)
ridge.fit(X_train_scaled, y_train)

# Lasso with a milder alpha so it doesn't kill 5/8 features
lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_train_scaled, y_train)

# ---- Evaluate ----
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)

print("=" * 54)
print(f"{'Model':<10} {'MSE':>10} {'R2':>8}")
print("-" * 54)
for name, yp in [("Ridge", y_pred_ridge), ("Lasso", y_pred_lasso)]:
    mse = np.mean((yp - y_test) ** 2)
    r2 = r2_score(y_test, yp)
    print(f"{name:<10} {mse:>10.4f} {r2:>8.4f}")
print("=" * 54)
# ---- Ridge sparsity ----
coef = ridge.coef_
n_total = len(coef)
n_nonzero = int(np.sum(np.abs(coef) > 1e-8))
n_zero = n_total - n_nonzero

print(
    f"\nRidge sparsity (alpha={ridge.alpha}): "
    f"{n_nonzero}/{n_total} features kept, "
    f"{n_zero}/{n_total} zeroed out"
)

order = np.argsort(np.abs(coef))[::-1]
print(f"\n{'Feature':<14} {'Coeff':>10}  {'Status'}")
print("-" * 38)
for idx in order:
    status = "-> 0 (dropped)" if abs(coef[idx]) < 1e-8 else "kept"
    print(f"{feature_names[idx]:<14} {coef[idx]:>10.4f}  {status}")

# ---- Lasso sparsity ----
coef = lasso.coef_
n_total = len(coef)
n_nonzero = int(np.sum(np.abs(coef) > 1e-8))
n_zero = n_total - n_nonzero

print(
    f"\nLasso sparsity (alpha={lasso.alpha}): "
    f"{n_nonzero}/{n_total} features kept, "
    f"{n_zero}/{n_total} zeroed out"
)

order = np.argsort(np.abs(coef))[::-1]
print(f"\n{'Feature':<14} {'Coeff':>10}  {'Status'}")
print("-" * 38)
for idx in order:
    status = "-> 0 (dropped)" if abs(coef[idx]) < 1e-8 else "kept"
    print(f"{feature_names[idx]:<14} {coef[idx]:>10.4f}  {status}")
