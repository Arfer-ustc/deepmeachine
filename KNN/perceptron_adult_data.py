import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

def standardize(X):
    X_std = np.zeros(X.shape)
    mean = np.mean(X, axis = 0)
    # X.mean(axis = 0)
    std = np.std(X, axis = 0)
    X_std = (X - mean)/std

    return X_std


data = pd.read_csv('adult.data')


print(data.columns)
# print(data.info())
print(data.head())
# print(data.describe())
# print(len(data[data[]]))

data.drop(['education', 'fnlwgt',  'capital_gain', 'capital_loss'], axis=1, inplace=True)

print(data.head())
print(data['income'].value_counts())
# <= 50K 24720
# > 50K 7841

# age_notnull = data[data["age"].notnull()].as_matrix()
# age_null = data[data["age"].isnull()].as_matrix()

# print(data["age"].isnull().value_counts())
print(data["type_employer"].value_counts()) #?
print(data["education_num"].value_counts())
print(data["marital"].value_counts())
print(data["occupation"].value_counts())# ?
print(data["relationship"].value_counts())
# print(data["race"].value_counts())
print(data["sex"].value_counts())
print(data["hr_per_week"].value_counts())
# print(data["country"].value_counts()) # ?

# print(data[data["type_employer"] == ' ?'].index.tolist())


# 删除缺失值记录
data.drop(data[data["type_employer"] == ' ?'].index.tolist(),axis=0, inplace=True)
data.drop(data[data["occupation"] == ' ?'].index.tolist(),axis=0, inplace=True)
print(data["type_employer"].value_counts())


print(data['income'].value_counts())
# 将离散值（类型值）改为数值

dummies_type_employer= pd.get_dummies(data["type_employer"], prefix = "type_employer")
dummies_marital = pd.get_dummies(data["marital"], prefix = "marital")
dummies_occupation = pd.get_dummies(data["occupation"], prefix = "occupation")
dummies_relationship = pd.get_dummies(data["relationship"], prefix = "relationship")
# dummies_race = pd.get_dummies(data["race"], prefix = "race")
# dummies_country = pd.get_dummies(data["country"], prefix = "country")
dummies_sex = pd.get_dummies(data["sex"], prefix = "sex")
print(dummies_sex, dummies_sex.shape)

data = pd.concat([data, dummies_type_employer, dummies_marital, dummies_occupation, dummies_relationship, dummies_sex], axis=1)
print(data.head())
data.drop(["type_employer", "marital", "occupation", "relationship", "sex"], axis=1, inplace=True)

print(data.head())

# 标准化

age_scale = standardize(np.array(data["age"]))
education_num_scale = standardize(np.array(data["education_num"]))
hr_per_week_scale = standardize(np.array(data["hr_per_week"]))
# print(data['age'].shape, age_scale.shape)

data["age_scale"] = age_scale
data["education_num_scale"] = education_num_scale
data["hr_per_week_scale"] = hr_per_week_scale


# data = pd.concat([data, pd.Series(age_scale), pd.Series(education_num_scale), pd.Series(hr_per_week_scale)], axis=1)
data.drop(["age", "education_num", "hr_per_week"], axis=1, inplace=True)

print(data.head())

data.loc[data['income'] == ' <=50K', 'income'] = -1
data.loc[data['income'] == ' >50K', 'income'] = 1

# print(data['age_scale'].shape, age_scale.shape)
# print(data['sex_ Female'].shape, dummies_sex.shape)


data.to_csv('adult_scale.data', index=False)




