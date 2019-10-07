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


model = linear_model.SGDClassifier(loss='log', learning_rate='constant', eta0=0.1,
                                   verbose=2, shuffle=True)

train_scores = []
test_scores = []

folds = 5
#NEED TO ASK HOW TO ACCESS THE VALUES TOGETHER
kf = KFold(folds, shuffle=True)
for train_i, test_i in kf.split(data):
    x_train = data.loc[train_i, 'number']
    x_test = data.loc[test_i, 'number']
    y_train = data.loc[train_i, 'y']
    y_test = data.loc[test_i, 'y']
    
    x_train = np.array(x_train).reshape(-1,1)
    x_test = np.array(x_test).reshape(-1,1)
    
    model.partial_fit(x_train, y_train, classes=[0,1])
    train_scores.append(model.score(x_train, y_train))
    test_scores.append(model.score(x_test, y_test))
    
    
#ASK ABOUT OUTPUT
a,b,c,d,e = kf.split(data)

    
train_errors = np.std(train_scores)
test_errors = np.std(test_scores)

print(f"Training/Testing errors: {train_errors}/{test_errors}")
epoch_array = np.arange(1,folds+1,1)

plt.errorbar(epoch_array, train_scores, yerr=train_errors)
plt.fill_between(epoch_array, train_scores-train_errors, train_scores+train_errors
                 ,alpha=0.2)
'''
plt.fill_between(epoch_array, train_scores-2*train_errors, train_scores+2*train_errors
                 ,alpha=0.2)
plt.fill_between(epoch_array, train_scores-3*train_errors, train_scores+3*train_errors
                 ,alpha=0.05)
'''
plt.errorbar(epoch_array, test_scores, yerr=test_errors)
plt.fill_between(epoch_array, test_scores-test_errors, test_scores+test_errors
                 ,alpha=0.2)

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()





    
    
    