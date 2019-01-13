from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def accuracy(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return np.sum(y==y_pred)/len(y)

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
 #   print("iris.target", iris.target)
 #   print("iris.data\n", iris.data)

    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:, :])
    return data[:, :2], data[:, -1]

#为方便后期画图更直观，故只取了前两列特征值向量训练

X, y = create_data()
#print("X~~~\n",X)
#print("y!!!\n",y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(X_train, y_train.ravel())

#kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
#kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
#decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
#decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。

print(clf.score(X_train, y_train))  # 精度
y_hat = clf.predict(X_train)

print ('训练集:\n',accuracy(y_hat, y_train))
print(clf.score(X_test, y_test))

y_hat = clf.predict(X_test)

print('测试集:\n',accuracy(y_hat, y_test))


#绘图
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()  # 第0列的范围
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# print 'grid_test = \n', grid_testgrid_hat = clf.predict(grid_test)       # 预测分类值grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

#mpl.rcParams['font.sans-serif'] = [u'SimHei']
#mpl.rcParams['axes.unicode_minus'] = False

#cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
#cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
#plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=50)  # 样本
plt.scatter(X_test[:, 0], X_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
# plt.grid()
plt.show()