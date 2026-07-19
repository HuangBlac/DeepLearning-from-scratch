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

# 4.1生成一个二维线性可分数据集。训练你的 LinearSVM，识别支持向量。验证支持向量正是离决策边界最近的那些点。
def generate_linear_data(N=100):
    # 生成一个二维线性可分数据集
    X, y = datasets.make_classification(n_samples= N , n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
    return X, y

X, y = generate_linear_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# make_classification 生成的是 {0, 1} 标签，而 SVM 需要 {-1, +1} 标签
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)
# 训练线性SVM
#定义hinge作为损失函数，实际max(0, 1 - margin)表示损失函数，当margin大于等于1时，损失函数为0，当margin小于1时，损失函数为1 - margin
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (np.dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
#定义Logistic Regression的损失函数，实际log(1 + exp(-z))表示损失函数，当z大于等于0时，损失函数为0，当z小于0时，损失函数为log(1 + exp(z))，log(1 + exp(z))是logistic函数的对数
def logistic_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        z = y[i] * (np.dot(w, X[i]) + b)
        total_loss += np.log(1 + np.exp(-z))
    return total_loss / n

class LinearSVM:
    def __init__(self, C=1.0):
        '''
        C: 正则化参数，C 越大 → 间隔越窄（过拟合倾向），C 越小 → 间隔越宽（欠拟合倾向）
        n_epochs: 训练轮数
        lr: 学习率
        '''
        self.C = C
        self.w = None
        self.b = None
        self.n_epochs = 1000
        self.lr = 0.01

    def fit(self, X, y):
        """
        最小化 L = C * Σ max(0, 1 - y_i(w·x_i + b)) + (λ/2)||w||²
        hinge loss 梯度逐样本更新，L2 正则化每个 epoch 施加一次。
        """
        n_samples = len(X)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.reg = 0.001  # L2 正则化系数 λ

        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin < 1:
                    # hinge loss 梯度: dL/dw = -C * y_i * x_i, dL/db = -C * y_i
                    self.w += self.lr * self.C * y[i] * X[i]
                    self.b += self.lr * self.C * y[i]
            # L2 正则化: d/dw (λ/2)||w||² = λ·w，每个 epoch 末施加一次
            self.w -= self.lr * self.reg * self.w
            if epoch % 100 == 0:
               print(f"Epoch {epoch + 1}/{self.n_epochs} completed.")
        print("Training completed.")
    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        # np.sign(0) = 0，但真实标签只有 -1 和 +1，将 0 归为正类
        return np.where(scores >= 0, 1, -1)

# 验证支持向量是距离最近的那个
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (np.dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors

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

    # 标出支持向量（离间隔边界最近的点）
    sv_idx = find_support_vectors(X, y, w, b)
    ax.scatter(X[sv_idx, 0], X[sv_idx, 1], s=100, facecolors='none',
               edgecolors='k', linewidths=1.5, label=f'SV ({len(sv_idx)})')

    # 数据点
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='o', label='Class -1')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s', label='Class +1')

    ax.set_xlabel('x₀'), ax.set_ylabel('x₁')
    ax.set_title(title)
    ax.legend(fontsize=7)


# ── 4.1 线性可分数据集 + 支持向量验证 ──
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
def generate_noisy_data(N=100):
    X, y = datasets.make_classification(n_samples=N, n_features=2, n_informative=2,
                                        n_redundant=0, n_classes=2, random_state=42)
    X += np.random.normal(0, 0.5, X.shape)
    return X, y


Xn, yn = generate_noisy_data()
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.2, random_state=42)
yn_train = np.where(yn_train == 0, -1, 1)
yn_test = np.where(yn_test == 0, -1, 1)

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
    
# 4.3造一个类边界是圆形（非线性）的数据集。说明线性 SVM 会失败。计算 RBF 核矩阵，说明在核诱导的特征空间里两类变得可分。
def data_circle():
    X, y = datasets.make_circles(n_samples=100, noise=0.1, random_state=42)
    return X, y

Xc, yc = data_circle()
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
yc_train = np.where(yc_train == 0, -1, 1)
yc_test = np.where(yc_test == 0, -1, 1)

model_linear = LinearSVM(C=1.0)
model_linear.fit(Xc_train, yc_train)
y_pred_linear = model_linear.predict(Xc_test)
print("Linear SVM Accuracy:", accuracy_score(yc_test, y_pred_linear))
print("Linear SVM Classification Report:\n", classification_report(yc_test, y_pred_linear, zero_division=0))
print("Linear SVM Confusion Matrix:\n", confusion_matrix(yc_test, y_pred_linear))

def rbf_kernel(X, Z, gamma=0.5):
    """计算 RBF 核矩阵 K_{ij} = exp(-gamma * ||X_i - Z_j||^2).
    利用恒等式 ||x - z||^2 = ||x||^2 + ||z||^2 - 2<x, z> 避免逐对循环。
    X: (n, d), Z: (m, d) → 返回: (n, m)
    """
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)   # (n, 1)
    Z_norm = np.sum(Z ** 2, axis=1).reshape(1, -1)    # (1, m)
    sq_dist = X_norm + Z_norm - 2 * np.dot(X, Z.T)     # (n, m)
    return np.exp(-gamma * sq_dist)


# 核矩阵特征维度高、信号弱，需要更大的学习率和更多轮次才能收敛
model_RBF = LinearSVM(C=1.0)
model_RBF.lr = 0.1
model_RBF.n_epochs = 5000
K_train = rbf_kernel(Xc_train, Xc_train, gamma=2.0)   # (n_train, n_train)
K_test  = rbf_kernel(Xc_test,  Xc_train, gamma=2.0)   # (n_test,  n_train)
model_RBF.fit(K_train, yc_train)
y_pred_RBF = model_RBF.predict(K_test)
print("RBF SVM Accuracy:", accuracy_score(yc_test, y_pred_RBF))
print("RBF SVM Classification Report:\n", classification_report(yc_test, y_pred_RBF, zero_division=0))
print("RBF SVM Confusion Matrix:\n", confusion_matrix(yc_test, y_pred_RBF))
# 4.4在同一数据集上对比 hinge loss 和 logistic loss。训练一个线性 SVM 和一个逻辑回归。数一数有多少训练点对各自模型的决策边界有贡献（支持向量 vs 全部点）。
