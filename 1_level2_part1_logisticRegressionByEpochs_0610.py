# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:08:06 2019

@author: Jack
"""

import sklearn
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_excel("numbers.xlsx")

x_train, x_test, y_train, y_test = train_test_split(data.number, data.y, 
                                                    train_size=0.9, test_size=0.1)

#NEED TO ASK HOW TO ACCESS THE VALUES TOGETHER


kf = KFold(6, shuffle=True)
for train_i, test_i in kf.split(data):
    x_train = data.loc[train_i, 'number']
    x_test = data.loc[test_i, 'number']
    y_train = data.loc[train_i, 'y']
    y_test = data.loc[test_i, 'y']
    
    


x_train = np.array(x_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)

model = linear_model.SGDClassifier(loss='log', learning_rate='constant', eta0=0.1,
                                   verbose=2, shuffle=True)


epochs = 10
train_scores = []
test_scores = []


for i in range(epochs):
    
    x,y = sklearn.utils.shuffle(x_train, y_train)
    model.partial_fit(x, y, classes=[0,1])
    train_scores.append(model.score(x, y))
    test_scores.append(model.score(x_test, y_test))
    
    
train_errors = np.std(train_scores)
test_errors = np.std(test_scores)

epoch_array = np.arange(1,11,1)

plt.errorbar(epoch_array, train_scores, yerr=train_errors)
plt.errorbar(epoch_array, test_scores, yerr=test_errors)
plt.show()





    
    
    