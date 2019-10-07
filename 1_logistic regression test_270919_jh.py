# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:55:24 2019


https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

@author: Jack
"""

import sklearn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


bc = datasets.load_breast_cancer()
half = int(len(bc.data)/2 + 0.5)
x = bc.data[:half]
x2 = pd.DataFrame(x)
x_1 = x2.loc[:,0]

y = bc.target[:half]

x_test = bc.data[half:]
y_test = bc.target[half:]

model = sklearn.linear_model.LogisticRegression()
model.fit(x_1,y)
pred = model.predict(x_test)
print(pred , " : ", y_test)

model.score(x_test, y_test)
df = pd.DataFrame(model.predict_proba(x_test))

x_nums = [i for i in range(len(x)-1)]
plt.scatter(x_nums, y_test)

