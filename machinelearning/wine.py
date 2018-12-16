# -*- coding: utf-8 -*-
# @Time    : 2018/12/15 下午6:37
# @Author  : xuef
# @FileName: pima-indians-diabetes.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_42118777/article
#Wine,Alcohol,Malic.acid,Ash,Acl,Mg,Phenols,Flavanoids,Nonflavanoid.phenols,Proanth,Color.int,Hue,OD,Proline
import csv
import random
import math


def loadCsv(filename):
    f=open(filename,"r")
    reader=csv.reader(f)
    dataset=list(reader)
    # lines = csv.reader(open(filename, "rb"))
    # dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

#SeparateByClass()函数按类别划分数据，然后计算出每个类的统计数据。
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[0] not in separated):
            separated[vector[0]] = []
        separated[vector[0]].append(vector)
    return separated

#计算在每个类中每个属性的均值、每个类中每个属性的标准差。均值是数据的中点或者集中 趋势，
# 在计算概率时，用来作为高斯分布的中值
def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    #删除第一个，因为第一个是结果，对属性特征的提取无用
    del summaries[0]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

#给定来自训练数据中已知属性的均值和标准差，可以使用高斯函数来评估一个给定的属性值 的概率
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

#合并一个数据样本中所有属性的概率，得到整个数据样本属于某个类的概率。
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#通过对测试数据集中每个数据样本的预测，我们可以评估模型精度。
# getPredictions()函数实 现这个功能，并返回每个测试样本的预测列表
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i][1:])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][0] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    filename = 'wine.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    # print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    # prepare model
    #summaries：{1:[[1,1,1,……],[1,1,1,……],[1,1,1,……]],0:[[],[],[]]}
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    # print('Accuracy: {0}%').format(accuracy)
    print("Accuracy:%f"%accuracy)


main()
