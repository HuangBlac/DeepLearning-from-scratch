'''
Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression tasks.
It works by finding the hyperplane that best separates the data points into different classes or predicts the output value for a given input.
在本格实验之中将会实现支持向量机这类算法，以及它常见的几类变体，核方法；还有SVR回归算法。
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# ═══════════════════════════════════════════════════════════════════════
# 模块级定义：类、函数
# ═══════════════════════════════════════════════════════════════════════

def generate_linear_data(N=100):
    """生成二维线性可分数据集。"""
    X, y = datasets.make_classification(
        n_samples=N, n_features=2, n_informative=2,
        n_redundant=0, n_classes=2, random_state=42)
    return X, y


def generate_noisy_data(N=100):
    """生成带噪声的二维线性可分数据集。"""
    X, y = datasets.make_classification(
        n_samples=N, n_features=2, n_informative=2,
        n_redundant=0, n_classes=2, random_state=42)
    X += np.random.normal(0, 0.5, X.shape)
    return X, y


def data_circle():
    """生成圆形（非线性）数据集。"""
    X, y = datasets.make_circles(n_samples=100, noise=0.1, random_state=42)
    return X, y


def rbf_kernel(X, Z, gamma=0.5):
    """计算 RBF 核矩阵 K_{ij} = exp(-gamma * ||X_i - Z_j||^2).
    利用恒等式 ||x - z||^2 = ||x||^2 + ||z||^2 - 2<x, z> 避免逐对循环。
    X: (n, d), Z: (m, d) → 返回: (n, m)
    """
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)   # (n, 1)
    Z_norm = np.sum(Z ** 2, axis=1).reshape(1, -1)    # (1, m)
    sq_dist = X_norm + Z_norm - 2 * np.dot(X, Z.T)     # (n, m)
    return np.exp(-gamma * sq_dist)


def generate_svr_data(N=100, seed=42):
    """生成 y = sin(x) + 噪声 的回归数据集。"""
    rng = np.random.RandomState(seed)
    X = np.linspace(0, 2 * np.pi, N).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, 0.1, N)
    return X, y


class LinearSVM:
    """线性 SVM，使用 hinge loss + L2 正则化，SGD 优化。
    损失: L = C * Σ max(0, 1 - y_i(w·x_i + b)) + (λ/2)||w||²
    """
    def __init__(self, C=1.0, lr=0.01, n_epochs=1000, reg=0.001):
        """
        C:        正则化参数，C 越大 → 间隔越窄（过拟合），C 越小 → 间隔越宽（欠拟合）
        lr:       学习率
        n_epochs: 训练轮数
        reg:      L2 正则化系数 λ
        """
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples = len(X)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin < 1:
                    # hinge loss 梯度: dL/dw = -C * y_i * x_i, dL/db = -C * y_i
                    self.w += self.lr * self.C * y[i] * X[i]
                    self.b += self.lr * self.C * y[i]
            # L2 正则化: d/dw (λ/2)||w||² = λ·w，每个 epoch 末施加一次
            self.w -= self.lr * self.reg * self.w

    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        # np.sign(0) = 0，但真实标签只有 -1 和 +1，将 0 归为正类
        return np.where(scores >= 0, 1, -1)


class SVR:
    """支持向量回归，使用 epsilon 不敏感损失 + L2 正则化。
    损失: L = C * Σ max(0, |y_i - f(x_i)| - ε) + (λ/2)||w||²
    """
    def __init__(self, C=1.0, epsilon=0.1, lr=0.01, n_epochs=2000, reg=0.001):
        self.C = C
        self.epsilon = epsilon
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.w = None
        self.b = None

    def fit(self, X, y):
        y = np.asarray(y).ravel()  # 确保 target 是一维
        n_samples = len(X)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                residual = y[i] - (np.dot(self.w, X[i]) + self.b)  # y - ŷ
                if residual > self.epsilon:
                    # 预测偏低 → 增加 w, b
                    self.w += self.lr * self.C * X[i]
                    self.b += self.lr * self.C
                elif residual < -self.epsilon:
                    # 预测偏高 → 减少 w, b
                    self.w -= self.lr * self.C * X[i]
                    self.b -= self.lr * self.C
                # |residual| ≤ ε → 点在管子内，无贡献
            # L2 正则化，每个 epoch 末施加一次
            self.w -= self.lr * self.reg * self.w

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def support_indices(self, X, y):
        """返回管子外的点（支持向量）的索引。"""
        residuals = np.abs(y.ravel() - self.predict(X))
        return np.where(residuals > self.epsilon)[0]


def find_support_vectors(X, y, w, b, tol=1e-6):
    """返回所有支持向量的索引（margin ≤ 1 的点，即落在间隔边界上或内部）。"""
    margins = y * (np.dot(X, w) + b)
    return np.where(margins <= 1.0 + tol)[0]


def plot_svm_boundary(ax, X, y, model, title=""):
    """在 2D 散点图上画出 SVM 的决策边界和间隔边界。"""
    # 构造网格
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 200),
                           np.linspace(x1_min, x1_max, 200))
    grid = np.c_[xx0.ravel(), xx1.ravel()]
    Z = model.predict(grid).reshape(xx0.shape)

    # 决策区域着色
    ax.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.coolwarm, levels=[-1, 0, 1])

    # 三条线：决策边界 (w·x + b = 0) 和间隔边界 (w·x + b = ±1)
    w, b = model.w, model.b
    x0_line = np.linspace(x0_min, x0_max, 100)
    if abs(w[1]) > 1e-6:
        x1_decision = -(w[0] * x0_line + b) / w[1]
        x1_margin_p = -(w[0] * x0_line + b - 1) / w[1]   # w·x + b = +1
        x1_margin_n = -(w[0] * x0_line + b + 1) / w[1]   # w·x + b = -1
        ax.plot(x0_line, x1_decision, 'k-', linewidth=2, label='Decision boundary')
        ax.plot(x0_line, x1_margin_p, 'k--', linewidth=1, label='Margin')
        ax.plot(x0_line, x1_margin_n, 'k--', linewidth=1)

    # 标出支持向量
    sv_idx = find_support_vectors(X, y, w, b)
    ax.scatter(X[sv_idx, 0], X[sv_idx, 1], s=100, facecolors='none',
               edgecolors='k', linewidths=1.5, label=f'SV ({len(sv_idx)})')

    # 数据点
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='o', label='Class -1')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s', label='Class +1')

    ax.set_xlabel('x₀'), ax.set_ylabel('x₁')
    ax.set_title(title)
    ax.legend(fontsize=7)


# ═══════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── 4.1 线性可分数据集 + 支持向量验证 ──
    X, y = generate_linear_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # make_classification 生成的是 {0, 1} 标签，而 SVM 需要 {-1, +1} 标签
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearSVM(C=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_svm_boundary(ax, X_train, y_train, model, title="Linear SVM (C=1.0) — Support Vectors")
    plt.tight_layout()
    plt.show()

    # ── 4.2 噪声数据 + 不同 C 值对比 ──
    Xn, yn = generate_noisy_data()
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.2, random_state=42)
    yn_train = np.where(yn_train == 0, -1, 1)
    yn_test = np.where(yn_test == 0, -1, 1)

    scaler_n = StandardScaler()
    Xn_train = scaler_n.fit_transform(Xn_train)
    Xn_test = scaler_n.transform(Xn_test)

    C_values = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, C in enumerate(C_values):
        svm_model = LinearSVM(C=C)
        svm_model.fit(Xn_train, yn_train)
        acc = accuracy_score(yn_test, svm_model.predict(Xn_test))
        plot_svm_boundary(axes[idx], Xn_train, yn_train, svm_model,
                          title=f"C={C} | acc={acc:.2f}")

    # 第 8 个子图留空
    for idx in range(len(C_values), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    # ── 4.3 圆形数据：线性 SVM vs RBF 核 SVM ──
    Xc, yc = data_circle()
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    yc_train = np.where(yc_train == 0, -1, 1)
    yc_test = np.where(yc_test == 0, -1, 1)

    # 线性 SVM（预期失败）
    model_linear = LinearSVM(C=1.0)
    model_linear.fit(Xc_train, yc_train)
    y_pred_linear = model_linear.predict(Xc_test)
    print("Linear SVM Accuracy:", accuracy_score(yc_test, y_pred_linear))
    print("Linear SVM Classification Report:\n",
          classification_report(yc_test, y_pred_linear, zero_division=0))
    print("Linear SVM Confusion Matrix:\n", confusion_matrix(yc_test, y_pred_linear))

    # RBF 核 SVM
    model_RBF = LinearSVM(C=1.0, lr=0.1, n_epochs=5000)
    K_train = rbf_kernel(Xc_train, Xc_train, gamma=2.0)
    K_test = rbf_kernel(Xc_test, Xc_train, gamma=2.0)
    model_RBF.fit(K_train, yc_train)
    y_pred_RBF = model_RBF.predict(K_test)
    print("RBF SVM Accuracy:", accuracy_score(yc_test, y_pred_RBF))
    print("RBF SVM Classification Report:\n",
          classification_report(yc_test, y_pred_RBF, zero_division=0))
    print("RBF SVM Confusion Matrix:\n", confusion_matrix(yc_test, y_pred_RBF))

    # ── 4.4 hinge loss vs logistic loss：支持向量 vs 全部点 ──
    X44, y44 = datasets.make_classification(
        n_samples=100, n_features=2, n_informative=2,
        n_redundant=0, n_classes=2, random_state=42)
    X44_train, X44_test, y44_train, y44_test = train_test_split(
        X44, y44, test_size=0.2, random_state=42)

    scaler44 = StandardScaler()
    X44_train = scaler44.fit_transform(X44_train)
    X44_test = scaler44.transform(X44_test)

    # SVM 需要 {-1, +1}，LR (sklearn) 使用 {0, 1}
    y44_train_svm = np.where(y44_train == 0, -1, 1)
    y44_test_svm  = np.where(y44_test == 0, -1, 1)

    svm44 = LinearSVM(C=1.0)
    svm44.fit(X44_train, y44_train_svm)

    lr44 = LogisticRegression()
    lr44.fit(X44_train, y44_train)

    print("\n=== Accuracy ===")
    print(f"SVM:  {accuracy_score(y44_test_svm, svm44.predict(X44_test)):.3f}")
    print(f"LR:   {accuracy_score(y44_test, lr44.predict(X44_test)):.3f}")

    # 核心对比：有多少训练点"推动"了决策边界
    # SVM: hinge loss 在 margin >= 1 时梯度严格为零 → 只有 margin < 1 的点有贡献
    svm_scores = np.dot(X44_train, svm44.w) + svm44.b
    svm_margins = y44_train_svm * svm_scores
    sv_count = np.sum(svm_margins <= 1 + 1e-6)

    # LR: logistic loss 梯度 dL_i/dw = -y_i * x_i * sigmoid(-y_i * score)
    # 梯度永远不为零，但远离边界的点贡献趋近于零
    lr_scores = lr44.decision_function(X44_train)
    lr_y = np.where(y44_train == 0, -1, 1)
    lr_z = lr_y * lr_scores
    lr_grad_norm = 1.0 / (1.0 + np.exp(lr_z))
    threshold = 1e-3
    lr_contributing = np.sum(lr_grad_norm > threshold)

    print("\n=== 对决策边界的贡献 ===")
    print(f"SVM: {sv_count}/{len(X44_train)} 个点推动决策边界（支持向量）")
    print(f"     其余 {len(X44_train) - sv_count} 个点梯度严格为零")
    print(f"LR:  {lr_contributing}/{len(X44_train)} 个点梯度范数 > {threshold}")
    print(f"     梯度范围: [{lr_grad_norm.min():.4f}, {lr_grad_norm.max():.4f}]")
    print("     → logistic loss 下所有点理论上都有非零梯度，与 hinge loss 本质不同")

    # ── SVR: epsilon 不敏感损失，拟合 y = sin(x) + 噪声 ──
    X_svr, y_svr = generate_svr_data()
    X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(
        X_svr, y_svr, test_size=0.2, random_state=42)

    # RBF 核 SVR
    K_train_svr = rbf_kernel(X_train_svr, X_train_svr, gamma=0.5)
    K_test_svr  = rbf_kernel(X_test_svr,  X_train_svr, gamma=0.5)

    model_svr = SVR(C=1.0, epsilon=0.2, lr=0.01, n_epochs=3000)
    model_svr.fit(K_train_svr, y_train_svr)

    y_pred_svr = model_svr.predict(K_test_svr)
    mse = np.mean((y_test_svr.ravel() - y_pred_svr) ** 2)
    print(f"SVR MSE: {mse:.4f}")

    # 画图：预测曲线 + ε 管子 + 支持向量
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_svr, y_svr, 'o', markersize=4, alpha=0.6, label='Data')

    X_dense = np.linspace(0, 2 * np.pi, 300).reshape(-1, 1)
    K_dense = rbf_kernel(X_dense, X_train_svr, gamma=0.5)
    y_dense = model_svr.predict(K_dense)

    ax.plot(X_dense, y_dense, 'r-', linewidth=2, label='SVR prediction')
    ax.fill_between(X_dense.ravel(),
                    y_dense - model_svr.epsilon,
                    y_dense + model_svr.epsilon,
                    alpha=0.15, color='red', label=f'ε-tube (ε={model_svr.epsilon})')

    sv_idx = model_svr.support_indices(K_train_svr, y_train_svr)
    ax.scatter(X_train_svr[sv_idx], y_train_svr[sv_idx].ravel(), s=60,
               facecolors='none', edgecolors='red', linewidths=1.5,
               label=f'SV ({len(sv_idx)})')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Kernel SVR on y = sin(x) + noise  |  MSE = {mse:.4f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
