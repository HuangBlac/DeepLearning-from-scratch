'''
Decision Tree
实现 Gini 不纯度、熵和信息增益的计算，来找出决策树的最优分裂
从零构建一个带预剪枝控制（最大深度、最小样本数）的决策树分类器
用自助采样和特征随机化构造一个随机森林，并解释它为什么能降低方差
对比 MDI 特征重要性和置换重要性，识别 MDI 何时会有偏
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.datasets import make_moons, make_classification

datasets = {
    "Iris": load_iris(return_X_y=True),
    "Wine": load_wine(return_X_y=True),
    "Breast Cancer": load_breast_cancer(return_X_y=True),
    "Moons": make_moons(n_samples=500, noise=0.3, random_state=42),
    "Synthetic": make_classification(n_samples=500, n_features=10, 
                                      n_classes=3, n_informative=5,
                                      n_redundant=2, random_state=42),
}


# 3-1:在一个有 3 个类的数据集上训练单棵决策树。手动追踪分裂，画出矩形决策边界。对比 max_depth=2 和 max_depth=10 时的边界。
#此处使用iris来作为数据集，有三个类，维度为2
data = make_blobs(n_samples=300, n_features=2, centers=3,
                  cluster_std=2.5, random_state=42)
X, y = data  # make_blobs 返回 (X, y) 元组，直接解包
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def gini_impurity(y):
    unique, counts = np.unique(y, return_counts=True)
    gini = 1 - np.sum((counts / len(y)) ** 2)
    return gini

def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    entropy = -np.sum((counts / len(y)) * np.log2(counts / len(y)))
    return entropy

def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) +
        (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity

class MyDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def _best_split(self, X, y):
        """遍历所有特征和切分点，找到信息增益最大的分裂方式"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        _, n_features = X.shape

        # 如果设置了 max_features，随机选特征子集
        feature_indices = list(range(n_features))
        if self.max_features is not None:
            feature_indices = list(
                np.random.choice(n_features, self.max_features, replace=False)
            )

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = information_gain(
                    y, y[left_mask], y[right_mask], self.criterion
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        """递归建树，返回一个节点（字典）"""
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))

        # 停止条件：深度达到上限 / 样本太少 / 只剩一类
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            return {"leaf": True, "class": np.argmax(np.bincount(y))}

        # 找最佳分裂
        feature, threshold, gain = self._best_split(X, y)

        # 找不到有效分裂时直接返回叶节点
        if gain <= 0:
            return {"leaf": True, "class": np.argmax(np.bincount(y))}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # 如果某一侧样本太少，不分裂
        if (
            left_mask.sum() < self.min_samples_leaf
            or right_mask.sum() < self.min_samples_leaf
        ):
            return {"leaf": True, "class": np.argmax(np.bincount(y))}

        # 累积特征重要性（不纯度减少量加权）
        self.feature_importances_[feature] += gain * n_samples

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[right_mask], y[right_mask], depth + 1),
        }

    def _predict_one(self, x, node):
        """对单个样本沿树向下走到叶节点，返回预测类别"""
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
    
    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree
        if node["leaf"]:
            print(f"{indent}Predict: class {node['class']}")
            return
        print(f"{indent}Feature {node['feature']} <= {node['threshold']:.4f}?")
        print(f"{indent}  Yes:")
        self.print_tree(node["left"], indent + "    ")
        print(f"{indent}  No:")
        self.print_tree(node["right"], indent + "    ")


def plot_decision_boundary(model, X, y, title):
    """对 2D 数据画出决策树矩形决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    Z = np.array(model.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    # 决策树的边界天然是轴对齐矩形 —— 用 pcolormesh 清晰展示
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis", levels=3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis",
                          edgecolor="k", s=40)
    plt.colorbar(scatter, ticks=[0, 1, 2], label="class")
    plt.title(title)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close()


model1 = MyDecisionTree(max_depth=2)
model2= MyDecisionTree(max_depth=10)
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
y1_predict = model1.predict(X_test)
y2_predict = model2.predict(X_test)
print("max_depth=2的准确率:", np.mean(y1_predict == y_test))
model1.print_tree()
plot_decision_boundary(model1, X, y, "Decision Tree (max_depth=2)")

print("max_depth=10的准确率:", np.mean(y2_predict == y_test))
model2.print_tree()
plot_decision_boundary(model2, X, y, "Decision Tree (max_depth=10)")

# 3-2:为回归树实现方差减少分裂。为 200 个点生成 y = sin(x) + 噪声，拟合你的回归树。把树的分段常数预测和真实曲线画在一起对比。
def data_gen(N = 200, noise=0.1):
    x = np.linspace(0, 2 * np.pi, N).reshape(-1, 1)
    y = np.sin(x).ravel() + np.random.normal(0, noise, N)
    return x, y
X1, Y1 = data_gen()
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X1,Y1,test_size = 0.2,random_state=42)
def variance_reduction(parent_labels, left_labels, right_labels):
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = np.var(parent_labels)
    child_impurity = (
        (n_left / n) * np.var(left_labels) +
        (n_right / n) * np.var(right_labels)
    )
    return parent_impurity - child_impurity

class MyRegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features = len(X)
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def _best_split(self, X, y):
        """遍历所有特征和切分点，找到信息增益最大的分裂方式"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        _, n_features = X.shape

        # 如果设置了 max_features，随机选特征子集
        feature_indices = list(range(n_features))
        if self.max_features is not None:
            feature_indices = list(
                np.random.choice(n_features, self.max_features, replace=False)
            )

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = variance_reduction(
                    y, y[left_mask], y[right_mask]
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        """递归建树，返回一个节点（字典）"""
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))

        # 停止条件：深度达到上限 / 样本太少 / 只剩一类
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            return {"leaf": True, "value": np.mean(y)}

        # 找最佳分裂
        feature, threshold, gain = self._best_split(X, y)

        # 找不到有效分裂时直接返回叶节点
        if gain <= 0:
            return {"leaf": True, "value": np.mean(y)}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # 如果某一侧样本太少，不分裂
        if (
            left_mask.sum() < self.min_samples_leaf
            or right_mask.sum() < self.min_samples_leaf
        ):
            return {"leaf": True, "value": np.mean(y)}

        # 累积特征重要性（不纯度减少量加权）
        self.feature_importances_[feature] += gain * n_samples

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[right_mask], y[right_mask], depth + 1),
        }

    def _predict_one(self, x, node):
        """对单个样本沿树向下走到叶节点，返回预测类别"""
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
    
    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree
        if node["leaf"]:
            print(f"{indent}Predict: class {node['value']}")
            return
        print(f"{indent}Feature {node['feature']} <= {node['threshold']:.4f}?")
        print(f"{indent}  Yes:")
        self.print_tree(node["left"], indent + "    ")
        print(f"{indent}  No:")
        self.print_tree(node["right"], indent + "    ")


Model_Regression = MyRegressionTree(max_depth=10)
Model_Regression.fit(X1_train,Y1_train)
y_predict = Model_Regression.predict(X1_test)
print("回归树的MSE:", np.mean((np.array(y_predict) - Y1_test) ** 2))

# 画图：真实 sin 曲线 vs 树的分段常数预测
X_full = np.linspace(0, 2 * np.pi, 500).reshape(-1, 1)
y_true = np.sin(X_full).ravel()
y_tree_pred = np.array(Model_Regression.predict(X_full))

plt.figure(figsize=(10, 5))
plt.plot(X_full, y_true, "b-", label="True sin(x)", linewidth=2)
plt.plot(X_full, y_tree_pred, "r--", label="Regression Tree (max_depth=3)", linewidth=1.5)
plt.scatter(X1_train, Y1_train, c="gray", s=20, alpha=0.5, label="Train data")
plt.scatter(X1_test, Y1_test, c="green", s=20, alpha=0.5, label="Test data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Regression Tree: Piecewise Constant Prediction vs sin(x)")
plt.legend()
plt.savefig("regression_tree_fit.png", dpi=150, bbox_inches="tight")
plt.close()

# 3-3 构建 1、5、10、50 和 200 棵树的随机森林。把训练准确率和测试准确率随树数量的变化画出来。观察到测试准确率会到平台期但不会下降（森林抗过拟合）。
X_iris, y_iris = load_iris(return_X_y=True)
X_Iris_train, X_Iris_test, y_Iris_train, y_Iris_test = train_test_split(X, y, random_state=42)

for n_estimators in {1,5,10,50,200}:
    rf = RandomForestClassifier(n_estimators, random_state=42)
    rf.fit(X_Iris_train, y_Iris_train)
    print(f"Accuracy: {rf.score(X_Iris_test, y_Iris_test):.4f}")
    
    print(f"Feature importances: {rf.feature_importances_}")
# 3-4 在 5 个不同数据集上对比用 Gini 不纯度和熵作为分裂准则。测量准确率和树深度。大多数情况下它们产出几乎相同的结果。解释为什么。
Dataset = {
    "Iris": load_iris(return_X_y=True),
    "Wine": load_wine(return_X_y=True),
    "Breast Cancer": load_breast_cancer(return_X_y=True),
    "Moons": make_moons(n_samples=500, noise=0.3, random_state=42),
    "Synthetic": make_classification(n_samples=500, n_features=10, 
                                      n_classes=3, n_informative=5,
                                      n_redundant=2, random_state=42),
}
for name, (X, y) in Dataset.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_gini = MyDecisionTree(criterion="gini", max_depth=5)
    model_ent = MyDecisionTree(criterion="entropy", max_depth=5)
    model_gini.fit(X_train, y_train)
    model_ent.fit(X_train, y_train)

    acc_gini = np.mean(model_gini.predict(X_test) == y_test)
    acc_ent = np.mean(model_ent.predict(X_test) == y_test)

    # 计算树深度：递归遍历
    def tree_depth(node):
        if node["leaf"]:
            return 0
        return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))

    print(f"--- {name} ---")
    print(f"  Gini:   accuracy={acc_gini:.4f}, depth={tree_depth(model_gini.tree)}")
    print(f"  Entropy: accuracy={acc_ent:.4f}, depth={tree_depth(model_ent.tree)}")

# 3-5 实现置换重要性。在一个特征是随机噪声但基数很高的数据集上，把它和 MDI 重要性对比。MDI 会把噪声特征排得很高，置换重要性不会。

# Step 1: 构造特征 —— 2 个有效特征 + 18 个噪声 + 5 个高基数随机列
rng = np.random.RandomState(42)
X_base, y = make_classification(
    n_samples=500, n_features=20, n_informative=2,
    n_redundant=3, n_clusters_per_class=1, n_classes=3, random_state=42
)
# 加 10 列高基数纯随机噪声（每个样本唯一）
noise_cols = rng.randn(500, 10)
X_with_noise = np.column_stack([X_base, noise_cols])

# 构造特征名：前 20 个 f0-f19，后 5 个 noise_0-noise_4
feature_names = [f"f{i}" for i in range(20)] + [f"noise_{i}" for i in range(10)]

X_train, X_test, y_train, y_test = train_test_split(
    X_with_noise, y, test_size=0.2, random_state=42
)

# Step 2: 训练决策树，读取 MDI 重要性
model_mdi = MyDecisionTree(max_depth=10, criterion="gini")
model_mdi.fit(X_train, y_train)
mdi_importances = model_mdi.feature_importances_

print("=== MDI 特征重要性 ===")
for name, imp in zip(feature_names, mdi_importances):
    if imp > 0.001:
        print(f"  {name:>12}: {imp:.4f}")

# 聚焦：MDI 给噪声列的均值和真实列的均值
noise_mdi = [imp for name, imp in zip(feature_names, mdi_importances) if "noise" in name]
real_mdi = [imp for name, imp in zip(feature_names, mdi_importances) if "noise" not in name]
print(f"\nMDI avg: real_feats={np.mean(real_mdi):.4f}, noise_feats={np.mean(noise_mdi):.4f}")

def permutation_importance(X, y, n_repeats=5, random_state=42):
    """置换重要性：逐个特征打乱，测量准确率下降"""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state)
    n_features = X.shape[1]
    rng_local = np.random.RandomState(random_state)

    model_base = MyDecisionTree(max_depth=10, criterion="gini")
    model_base.fit(X_tr, y_tr)
    base_acc = np.mean(model_base.predict(X_te) == y_te)

    importance_scores = np.zeros(n_features)
    for feature_idx in range(n_features):
        print(f"  特征 {feature_idx}/{n_features}...", end=" ", flush=True)
        scores = []
        for _ in range(n_repeats):
            X_shuffled = X_tr.copy()
            rng_local.shuffle(X_shuffled[:, feature_idx])
            model = MyDecisionTree(max_depth=10, criterion="gini")
            model.fit(X_shuffled, y_tr)
            shuff_acc = np.mean(model.predict(X_te) == y_te)
            scores.append(base_acc - shuff_acc)
        importance_scores[feature_idx] = np.mean(scores)
    return importance_scores


# Step 3: 计算置换重要性并画图对比
print("\n计算置换重要性 (30 features × 3 repeats = 90 trees)...")
perm_importances = permutation_importance(X_train, y_train, n_repeats=3)

perm_noise = [imp for name, imp in zip(feature_names, perm_importances) if "noise" in name]
perm_real = [imp for name, imp in zip(feature_names, perm_importances) if "noise" not in name]
print(f"\nPermutation avg: real_feats={np.mean(perm_real):.4f}, noise_feats={np.mean(perm_noise):.4f}")

# 并排柱状图
x = np.arange(len(feature_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, mdi_importances, width, label="MDI (biased)", color="steelblue")
ax.bar(x + width/2, perm_importances, width, label="Permutation (unbiased)", color="darkorange")

ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Importance Score")
ax.set_title("MDI vs Permutation Importance:\nNoise features get high MDI but ~0 Permutation")
ax.legend()
ax.axhline(y=0, color="gray", linewidth=0.5)
plt.tight_layout()
plt.savefig("mdi_vs_permutation.png", dpi=150, bbox_inches="tight")
plt.close()