# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 21:28:29 2019

@author: Jack
"""


import sklearn
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing

data = pd.read_excel("numbers.xlsx")

x_train, x_test, y_train, y_test = train_test_split(data.number, data.y, 
                                                    train_size=0.9, test_size=0.1)

x_train = np.array(x_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)


#SVM's aren't scale invariant so have to normalise?

'''
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
'''


g = np.arange(0.01,1.5,0.05)
model_scores = []
for gamma in g:
    model = svm.SVC(verbose=2, gamma=gamma)
    model.fit(x_train, y_train)
    
    a = model.score(x_test, y_test)
    model_scores.append(a)
    
    
plt.plot(g,model_scores)
plt.xlabel("Gamma")
plt.ylabel("Accuracy")

#Doesn't make sense to use negative gamma values because the gamma is the
#width of the gaussian - cannot have negative width!
