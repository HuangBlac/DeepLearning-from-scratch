'''
在这一部分之中，我们主要去了解logistic回归这种算法
它是在线性回归的基础上叠加上一个函数\sigma，通过将函数区间压缩到[0,1]从而实现对概率问题进行理解
'''

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
#构建一个线性不可分的同心圆数据集，然后使用Logistic回归进行分类
def data_generate(N = 100,R1 = 1,R2 =2,expand = False):
    #生成两个同心圆，小圆半径为R1，大圆半径为R2，观测到的变量包括x1,x2,expand表示是否需要进行是否将特征扩展为多项式x1^2,x2^2,x1x2
    theta = np.random.uniform(0,2*np.pi,N//2)
    r1 = R1 + np.random.uniform(0,0.1,N//2)
    r2 = R2 + np.random.uniform(0,0.1,N//2)
    x1 = np.column_stack((r1*np.cos(theta),r1*np.sin(theta)))
    x2 = np.column_stack((r2*np.cos(theta),r2*np.sin(theta)))
    x = np.vstack((x1,x2))
    y = np.hstack((np.zeros(N//2),np.ones(N//2)))
    if expand:
        x = np.hstack((x,np.power(x,2),(x[:,0]*x[:,1]).reshape(-1,1)))
    return x,y

#定义Logistic回归函数类
class MyLogisticRegression:
    def __init__(self,n_features,alpha = 0.01):
        self.alpha = alpha
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []
    def prob(self,x):
        p = 1/(1+np.exp(-np.dot(x,self.w)-self.b))
        return p
    
    def predict(self,x):
        p = self.prob(x)
        return np.where(p>=0.5,1,0)
    
    def loss(self,x,y):
        y_pred = self.prob(x)
        N = len(y)
        loss = -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/N
        return loss
    
    def fit(self,x,y):
        for _ in range(1000):
            loss = self.loss(x,y)
            p = self.prob(x)
            grad_w = np.dot(x.T,(p-y))/len(y)
            grad_b = np.sum(p-y)/len(y)
            self.w -= self.alpha*grad_w
            self.b -= self.alpha*grad_b
            self.loss_history.append(loss)
        return self.w,self.b
    
    def accuracy(self,x,y):
        y_pred = self.predict(x)
        return accuracy_score(y,y_pred)
    
X_0,y_0 = data_generate(N = 100,R1 = 1,R2 = 2,expand = False)
X0_train,X0_test,y0_train,y0_test = train_test_split(X_0,y_0,test_size = 0.2,random_state = 42)
X_1,y_1 = data_generate(N = 100,R1 = 1,R2 = 2,expand = True)
X1_train,X1_test,y1_train,y1_test = train_test_split(X_1,y_1,test_size = 0.2,random_state = 42)

#使用Logistic回归进行分类
## 使用无扩展的情况作为特征
logistic = MyLogisticRegression(n_features = X0_train.shape[1])
logistic.fit(X0_train,y0_train)
y0_pred = logistic.predict(X0_test)
print("Accuracy:",logistic.accuracy(X0_test,y0_test))
## 加入了二次项的回归
logistic = MyLogisticRegression(n_features = X1_train.shape[1])
logistic.fit(X1_train,y1_train)
y1_pred = logistic.predict(X1_test)
print("Accuracy:",logistic.accuracy(X1_test,y1_test))

'''
第二题
以鸢尾花四特征，三分类问题为例子
使用Logistic回归和Softmax模型实现一个多分类混淆矩阵。计算每个类的精确率和召回率。哪个类最难分？
'''
data = load_iris()
X,y = data.data,data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 60)
#数据处理，对于y进行one-hot编码，对x进行标准化scaler标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Logistic回归
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
y_pred = logistic.predict(X_test)
#计算混淆矩阵
conf_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",conf_matrix)
print(classification_report(y_test, y_pred, target_names=data.target_names))
'''
从零画一条 ROC 曲线。
对从 0 到 1 的 100 个阈值，计算真正例率和假正例率。
用梯形法则算出 AUC（曲线下面积）。
'''
np.random.seed(42)
#生成示例数据集
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [3, 3]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = LogisticRegression()
lr.fit(X_tr_sc, y_tr)
y_pr = lr.predict(X_te_sc)
fpr, tpr, thresholds = roc_curve(y_te, y_pr)
roc_auc = auc(fpr, tpr)
print("AUC:",roc_auc)

