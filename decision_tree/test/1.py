# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 上午11:24
# @Author  : xuef
# @FileName: 1.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_42118777/article
from math import log
import operator

def credataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

dataset, labels = credataSet()

feat = [example[1] for example in dataset]

#计算信息熵
def calEnt(dataset):
    numEnteries = len(dataset)
    # 新建一个空的字典，这种用法通常用于数据集中字段计数
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Ent = 0.0
    for key in labelCounts:
        # 先转换为浮点
        prob = float(labelCounts[key]) / numEnteries
        Ent -= prob * log(prob, 2)
    return Ent

ent = calEnt(dataset)

# 按照指定特征划分数据集
# dataset：待划分的数据集，axis：划分数据集特征
# value：返回的特征的值
def splitDataSet(dataset, axis, value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

splitdata = splitDataSet(dataset, 0, 1)

def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calEnt(dataset)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        # 取每一个样本的第i+1个元素，featList = [1, 1, 0, 1, 1]
        # set(featList) = {0,1}，set是得到列表中唯一元素值的最快方法
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 遍历特征下属性值，对每一个属性值划分一次数据集
            subDataSet = splitDataSet(dataset, i, value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob * calEnt(subDataSet)
        # 计算两个特征各自的信息增益，并进行比较
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

bestfeature = chooseBestFeatureToSplit(dataset)

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据的长度为1，那么意味着所有的特征都用完了，分类完毕并返回次数最多的分类名称
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    # 当dataset的特征被选中为bestFeat后便不再作为候选划分特征，所以要删除掉
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), subLabels)
    return myTree

createTree(dataset, labels)