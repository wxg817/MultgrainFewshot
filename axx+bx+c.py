# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 14:49:51 2023

@author: Administrator
"""

import numpy as np

# 提供的数据点
x = np.array([20, 120])
y = np.array([0.748, 0.8432])

# 使用2阶多项式拟合数据
coefficients = np.polyfit(x, y, 2)

# 获取拟合的系数
a, b, c = coefficients

print("a =", a)
print("b =", b)
print("c =", c)