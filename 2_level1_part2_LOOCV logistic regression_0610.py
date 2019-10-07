# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:54:54 2019

@author: Jack
"""

import sklearn
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_excel("numbers.xlsx")

loo = LeaveOneOut()

model = linear_model.SGDClassifier(loss='log', learning_rate='constant', eta0=0.1,
                                   verbose=2, shuffle=True, max_iter=4, tol=None)

train_errors = []
test_errors = []

for train_i, test_i in loo.split(data):
    
    x_train, x_test = data.loc[train_i, 'number'], data.loc[test_i, 'number']
    y_train, y_test = data.loc[train_i, 'y'], data.loc[test_i, 'y']
    x_train = np.array(x_train).reshape(-1,1)
    x_test = np.array(x_test).reshape(-1,1)
    
    model.fit(x_train, y_train)
    train_errors.append(model.score(x_train, y_train))
    test_errors.append(model.score(x_test, y_test))
    
mean_train_e = sum(train_errors)/len(train_errors)
mean_test_e = sum(test_errors)/len(test_errors)

print(f"Train: {mean_train_e}  Test: {mean_test_e}")
    
    
