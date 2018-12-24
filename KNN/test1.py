import copy
from matplotlib import pyplot as pl
from matplotlib import animation as ani
import csv
import numpy as np
import random
import logging
import pandas as pd
def _read_data():

    url='./adult.data'
    Adult = pd.read_csv(url, header=None)
    # Adult.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    Adult.drop(['education', 'fnlwgt', 'race', 'capital_gain', 'capital_loss', 'country'], axis=1, inplace=True)
    print(Adult.head())
    return Adult


def main():
    _read_data()
    # data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    #                    names=["age", "type_employer", "fnlwgt", "education", "education_num",
    #                           "marital", "occupation", "relationship", "race", "sex", "capital_gain",
    #                           "capital_loss", "hr_per_week", "country", "income"])
    #
    # data.to_csv('adult.data', index=False)



if __name__ == '__main__':
    main()