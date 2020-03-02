# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:31:34 2019

@author: Jack
"""

import numpy as np

coin_flips = 500

total = 0
for i in range(coin_flips):
    flip = np.random.rand()
    result = (1 if flip > 0.5 else 0)
    total += result
    
print(total / coin_flips, " = fraction of heads")