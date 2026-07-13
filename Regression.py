'''
Linear Regression
使用sklearn库实现线性回归算法来线性回归算法的实现
包括算法的实现与可视化的内容
'''
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
#定义数据集,包括特征和标签
random.seed(42)
#定义数据集0
def generate_toy_data(N = 100):
    X = np.random.uniform(0, 10, N)
    y = 5 * X + 7+ np.random.normal(0, 2.0, N)
    return X, y
#生成数据集1
def generate_data(N = 100, epsilon=0.1, a=2, b=1, c=1.5, d=0):
    X = np.random.uniform(0, 10, N)
    y = a * X**3 + b * X**2 + c * X + d + np.random.normal(0, epsilon, N)
    return X, y
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # sklearn needs 2D features
    X_train = np.asarray(X_train).reshape(-1, 1)
    X_test = np.asarray(X_test).reshape(-1, 1)
    return X_train, X_test, y_train, y_test
#实现线性回归算法
#手动实现的线性回归算法
class MyLinearRegression:
    '''
    Linear regression with batch GD, SGD, and mini-batch GD.
    Tracks cost history for convergence comparison.
    '''
    def __init__(self, w=0.0, b=0.0, alpha=0.01, max_iter=1000):
        self.w = w
        self.b = b
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_history = []          # MSE per epoch / per-sample update

    def predict(self, X):
        return self.w * np.asarray(X) + self.b

    def mse(self, X, y):
        yp = self.predict(X)
        return float(np.mean((yp - y) ** 2))

    def fit(self, X, y, method='sgd', batch_size=8):
        '''
        method: 'batch' | 'sgd' | 'minibatch'
        cost_history records an entry after EVERY parameter update.
        For 'sgd' that is every sample; for 'batch' every epoch.
        '''
        X = np.asarray(X); y = np.asarray(y)
        n = len(X)
        self.w, self.b = 0.0, 0.0
        self.cost_history = []

        if method == 'batch':
            for _ in range(self.max_iter):
                yp = self.predict(X)
                dw = 2 * np.mean((yp - y) * X)
                db = 2 * np.mean(yp - y)
                self.w -= self.alpha * dw
                self.b -= self.alpha * db
                self.cost_history.append(self.mse(X, y))

        elif method == 'sgd':
            for _ in range(self.max_iter):
                for i in np.random.permutation(n):          # shuffle each epoch
                    xi, yi = X[i], y[i]
                    yp = self.predict(xi)
                    dw = 2 * (yp - yi) * xi
                    db = 2 * (yp - yi)
                    self.w -= self.alpha * dw
                    self.b -= self.alpha * db
                    self.cost_history.append(self.mse(X, y))

        elif method == 'minibatch':
            for _ in range(self.max_iter):
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    batch_idx = idx[start:start + batch_size]
                    xb, yb = X[batch_idx], y[batch_idx]
                    yp = self.predict(xb)
                    dw = 2 * np.mean((yp - yb) * xb)
                    db = 2 * np.mean(yp - yb)
                    self.w -= self.alpha * dw
                    self.b -= self.alpha * db
                    self.cost_history.append(self.mse(X, y))

        return self.w, self.b


# ============================================================
# Compare batch GD, SGD, and mini-batch GD on the same dataset.
# ============================================================
# Generate data: y = 5x + 7 + noise
X0, y0 = generate_toy_data(N=100)
X_train0, X_test0, y_train0, y_test0 = train_test_split(
    X0, y0, test_size=0.2, random_state=42
)

EPOCHS = 50
history = {}
results = {}

for method in ('batch', 'sgd', 'minibatch'):
    model = MyLinearRegression(w=0, b=0, alpha=0.01, max_iter=EPOCHS)
    model.fit(X_train0, y_train0, method=method, batch_size=8)
    yp = model.predict(X_test0)
    r2 = r2_score(y_test0, yp)
    results[method] = r2
    history[method] = model.cost_history
    print(f"{method:<10}  R2={r2:.4f}  final MSE={model.cost_history[-1]:.4f}  "
          f"cost evals={len(model.cost_history)}")

# ---- Convergence analysis ----
# "Fastest": lowest cost after the same number of *epochs*
# Measure cost at the end of each epoch for fair comparison.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: cost per epoch (fair comparison)
ax = axes[0]
for method in ('batch', 'sgd', 'minibatch'):
    if method == 'sgd':
        # sgd records every sample-update; downsample to per-epoch mean
        n_train = len(X_train0)
        h = history[method]
        epoch_cost = [np.mean(h[i * n_train:(i + 1) * n_train]) for i in range(EPOCHS)]
        ax.plot(range(1, EPOCHS + 1), epoch_cost, marker='.', label=f'{method} (per-epoch avg)')
    elif method == 'minibatch':
        # minibatch records every batch-update
        n_train = len(X_train0)
        h = history[method]
        batches_per_epoch = max(1, n_train // 8)
        epoch_cost = [np.mean(h[i * batches_per_epoch:(i + 1) * batches_per_epoch])
                      for i in range(EPOCHS)]
        ax.plot(range(1, EPOCHS + 1), epoch_cost, marker='.', label=f'{method} (per-epoch avg)')
    else:
        ax.plot(range(1, EPOCHS + 1), history[method], marker='.', label=method)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.set_title('Cost vs Epoch (fair comparison)')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: raw cost history — SGD shows noisy descent
ax = axes[1]
for method in ('batch', 'sgd', 'minibatch'):
    ax.plot(history[method], alpha=0.7, linewidth=0.6, label=method)
ax.set_xlabel('Update step')
ax.set_ylabel('MSE')
ax.set_title('Raw cost per update (SGD noise visible)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gd_convergence_comparison.png', dpi=150)
print("\nFigure saved to gd_convergence_comparison.png")
print("Answer: batch GD converges smoothest; SGD converges fastest per epoch "
      "but is noisy; mini-batch balances both.")

# ================================================================
# Polynomial regression: fit degrees 1, 3, 10 to cubic data.
# Compare train R2 vs test R2 — at which degree does overfitting appear?
# ================================================================
# True model: y = 2x^3 + x^2 + 1.5x + noise
# Small N + moderate noise so that overfitting becomes visible
X1, y1 = generate_data(N=30, epsilon=30.0, a=2, b=1, c=1.5, d=0)
X_train1, X_test1, y_train1, y_test1 = split_data(X1, y1)

print("\n" + "=" * 62)
print(f"{'Degree':<10} {'Train R2':>10} {'Test R2':>10} {'Gap':>8}  Note")
print("-" * 62)

for degree in (1, 3, 10):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train1)
    X_test_poly = poly.transform(X_test1)

    # Scale to avoid huge values blowing up degree-10 fit
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly = scaler.transform(X_test_poly)

    # Fit and evaluate
    model = LinearRegression()
    model.fit(X_train_poly, y_train1)

    r2_train = r2_score(y_train1, model.predict(X_train_poly))
    r2_test = r2_score(y_test1, model.predict(X_test_poly))
    gap = r2_train - r2_test

    note = ""
    if degree == 1:
        note = "<-- underfit (high bias)"
    elif degree == 3:
        note = "<-- true model (~optimal)"
    elif gap > 0.02:
        note = "<-- OVERFIT (high variance)"

    print(f"{degree:<10} {r2_train:>10.4f} {r2_test:>10.4f} {gap:>8.4f}  {note}")

print("=" * 62)
print("Answer: degree=1 underfits (both R2 low, high bias).")
print("degree=3 matches the true process — highest test R2, small gap.")
print("degree=10 overfits — train R2 ~1.0 while test R2 drops;")
print("the widening gap shows the model is memorizing noise (high variance).")

# Visualize the three fits
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
x_curve = np.linspace(0, 10, 300).reshape(-1, 1)

for ax, degree in zip(axes, (1, 3, 10)):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(poly.fit_transform(X_train1))
    model = LinearRegression()
    model.fit(X_train_poly, y_train1)

    X_curve_poly = scaler.transform(poly.transform(x_curve))
    y_curve = model.predict(X_curve_poly)

    ax.scatter(X_train1, y_train1, s=15, alpha=0.4, label='train')
    ax.scatter(X_test1, y_test1, s=15, alpha=0.4, label='test')
    ax.plot(x_curve, y_curve, 'r-', lw=2, label=f'deg={degree}')
    ax.set_title(f"Degree {degree}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('poly_regression_comparison.png', dpi=150)
print("\nFigure saved to poly_regression_comparison.png")