# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:23:12 2019

Code to learn how to implement a logistic regression.

Code follows the tutorial at:
"https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8"

@author: Jack
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("banking.csv")
education_unique = list(data['education'].unique())


data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

data['y'].value_counts()

sns.countplot(x='y',data=data, palette='hls')
plt.show()

count_no_sub = len(data[data['y'] == 0])
count_sub = len(data[data['y'] == 1])
total = count_no_sub + count_sub
print(count_no_sub/total, "compared to ", count_sub/total)

y_mean = pd.DataFrame(data.groupby('y').mean())
job_mean = pd.DataFrame(data.groupby(['y','job']).mean())


pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

table = pd.crosstab(data.marital, data.y)
table = table.div(table.sum(1).astype(float), axis=0)
table.plot(kind='bar', stacked=True)


