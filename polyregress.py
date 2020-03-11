#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read the dataset into a dataframe
df = pd.read_excel('./curve.xlsx')

# Place the ranges in a variable and preprocess
x = df['x'].values
y = df['y'].values
x = x.reshape(-1, 1)

# Change the order here. degree is same as M
poly = PolynomialFeatures(degree=11)

# Fit a Polynomial Curve
X_poly = poly.fit_transform(x)
poly.fit(X_poly, y)
linreg = LinearRegression()
linreg.fit(X_poly, y)
y_pred = linreg.predict(X_poly)

# print(y_pred)

# print(linreg.intercept_)
# print(linreg.coef_)

# Plot the curves. The regression line is in red
plt.scatter(x,y, color='blue')
plt.plot(x, y_pred, color='red')
plt.show()

