'''
波士顿多特征房价数据集，使用线性回归模型进行预测
'''
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# 加载波士顿房价数据集
data = sklearn.datasets.load_boston()
X = data.data
y = data.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征工程，对特征进行标准化处理
X_train_scaled = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test_scaled = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# 创建线性回归模型
model1 = Ridge(alpha=0.1)
model2 = Lasso(alpha=0.1)

# 训练模型
model2.fit(X_train_scaled, y_train)
y_pred = model2.predict(X_test_scaled)

# 计算模型的均方误差
r2 = np.mean((y_pred - y_test) ** 2)

print(f"parameters of Lasso: {model2.coef_[0]:.4f}")
print(f"Ridge R-squared: {r2_score(y_test, model2.predict(X_test_scaled)):.4f}")
print(f"Ridge coefficient: {model2.coef_[0]:.4f}")
