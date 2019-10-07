# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:44:11 2019

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

x = np.array(data['number'])
y = np.array(data['y'])


def get_Errors(x,y,t):
    num_errors = 0
    try:
        for i in range(len(x)):
            f = x[i] - t
            if f >= 0:
                y_pred = 1
            else:
                y_pred = 0
            num_errors += abs(y_pred - y[i])
    except: 
        return None
    return num_errors
        

def decision_stump(x,y):
    
    xmin = x.min()
    xmax = x.max()
    step = 1
    min_errors = 15
    tolerance = 5
    threshold = None

    for i in range(xmin,xmax,step):
        
        number_of_errors = get_Errors(x,y,i)
        if number_of_errors < min_errors + tolerance:
            if (threshold != None) and get_Errors(x,y,threshold) > number_of_errors:
                threshold = i
            elif (threshold == None):
                threshold = i
            else:
                pass     
    if threshold == None:
        print(f"No stump found with less errors than {min_errors + tolerance}.")
    return threshold
            
    
threshold_val = decision_stump(x,y)    
errors = get_Errors(x,y,threshold_val)
plt.scatter(x,y)
plt.plot([threshold_val]*2,[0,1])
plt.show()
print("Number of errors: ", errors)


