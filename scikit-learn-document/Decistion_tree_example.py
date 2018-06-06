from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold
import numpy as np

##加载数据
iris = load_iris()
##设置筛选阀值
sel = VarianceThreshold(threshold=(.4 * (1 - .4)))

##设置训练集和标签集
X, y = iris.data, iris.target
print("原始数据：")
print(X.shape)  # (150, 4)

# 筛选数据
X_new = sel.fit_transform(X)
print("新数据集：")
print(X_new.shape)  # (150, 3)

##设置分类函数：决策树
clf = tree.DecisionTreeClassifier()
##训练数据
clf.fit(X, y)
##预测数据
y_pred = clf.predict(X)

##使用选择特征后的数据进行训练
clf.fit(X_new, y)
##在新数据集上进行预测
y_pred1 = clf.predict(X_new)

##原始数据的预测结果和真实结果的对比
cnt = 0
for i in range(len(y)):
    if y_pred[i] == y[i]:
        cnt += 1
print("原始数据的预测结果和真实结果相同的个数：")
print(cnt)

##新数据集和真实结果的对比
cnt = 0
for i in range(len(y)):
    if y_pred1[i] == y[i]:
        cnt += 1
print("新数据集的预测结果和真实结果相同的个数：")
print(cnt)

##原始数据的预测结果和新数据的预测结果的对比
cnt = 0
for i in range(len(y)):
    if y_pred[i] == y_pred1[i]:
        cnt += 1
print("原始据集的预测结果和新数据集预测结果相同的个数：")
print(cnt)
