import numpy as np
import pandas as pd
from sklearn import datasets
# from __future__ import print_function
import os
import sys
import math
import  matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    print('idx.shape: ', idx.shape)
    np.random.shuffle(idx)

    return X[idx], y[idx]

# 正规化数据集
def normalize(X, axis = -1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)

# 标准化数据集
def standardize(X):
    X_std = np.zeros(X.shape)
    mean = np.mean(X, axis = 0)
    # X.mean(axis = 0)
    std = np.std(X, axis = 0)

    for col in range(X.shape[1]):
        if std[col]:
            X_std[:, col] = (X[:, col] - mean[col]) / std[col]

    return X_std

#数值归一化
def autoNorm(X):
    minVals = X.min(0)
    maxVals = X.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(X.shape)
    m = X.shape[0]
    normDataSet = X - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def train_test_split(X, y, test_size = 0.2, shuffle = True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    n_train_samples = int(X.shape[0]*(1-test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return x_train, x_test, y_train, y_test

def accurary(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return np.sum(y==y_pred)/len(y)

class KNN():
    '''
    K近邻分类算法
    '''

    def __init__(self, k=5):
        self.k = k
    
    def euclidean_distance(self, one_sample, X_train):
        one_sample = one_sample.reshape(1, -1)
        X_train = X_train.reshape(X_train.shape[0], -1)
        distances = np.power(np.tile(one_sample, (X_train.shape[0], 1)) - X_train, 2).sum(axis = 1)

        return distances

    def get_k_neighbor_abels(self, distances, y_train, k):
        k_neighbor_labels = []
        for distance in np.sort(distances)[:k]:
            label = y_train[distances == distance]
            # print(type(label), label.shape)
            if label.shape[0] > 1:
                label = label.tolist()
                k_neighbor_labels.extend(label)
            else:
                k_neighbor_labels.append(label)

        return np.array(k_neighbor_labels).reshape(-1, )

    def vote(self, one_sample, X_train, y_train, k):
        distances = self.euclidean_distance(one_sample, X_train)
        # print('distances.shape: ', distances.shape)
        y_train = y_train.reshape(y_train.shape[0], 1)
        # print('y_train.shape: ', y_train.shape)
        k_neighbor_labels = self.get_k_neighbor_abels(distances, y_train, k)
        # print(k_neighbor_labels.shape, type(k_neighbor_labels))
        find_label, find_count = 0, 0
        k_neighbor_labels = list(k_neighbor_labels)
        for label, count in Counter(k_neighbor_labels).items():
            if count > find_count:
                find_count = count
                find_label = label
        return find_label

    def predict(self, X_test, X_train, y_train):
        y_pred = []
        for sample in X_test:
            label= self.vote(sample, X_train, y_train, self.k)
            y_pred.append(label)
        return np.array(y_pred)


def main():
    # data = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2)
    # X, y = data[0], data[1]
    data = iris.data
    target = iris.target
    # print(type(data), data.shape, type(target), target.shape)
    X, y = data, target

    # print(type(X), type(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle= True)
    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)
    accu = accurary(y_test, y_pred)

    print("Accurary: ", accu)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = main()
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('sklearn--result: ', acc)
