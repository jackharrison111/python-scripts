# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:50:48 2019

@author: Jack
"""

import sklearn
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_excel("numbers.xlsx")

x_train, x_test, y_train, y_test = train_test_split(data.number, data.y, 
                                                    train_size=0.7, test_size=0.3)
x_train = np.array(x_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)

data = {'x_train' : x_train, 'y_train' : y_train, 'x_test': x_test, 'y_test' : y_test}

def use_linear_model(eta0, data: dict):
    print(eta0)
    model = linear_model.SGDClassifier(verbose=1, loss='log', learning_rate='constant',
                                   eta0=eta0)
    model.fit(data['x_train'], data['y_train'])
    test_e = model.score(x_test, y_test)
    return test_e


#c = model.intercept_
#w = model.coef_
#train_e = model.score(x_train, y_train)
    
eta_vals = np.arange(0.0025,0.1,0.005)
performances = []
for val in eta_vals:
    error = use_linear_model(val, data)
    x = (val,error)
    performances.append(x)
    
x = [val[0] for val in performances]
y = [val[1] for val in performances]
plt.plot(x,y, marker='x')
plt.show()




'''
plt.scatter(x_train, y_train)
y = w*x_train + c
plt.plot(x_train, y)
plt.show()
'''