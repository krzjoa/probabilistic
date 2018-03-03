# Bayesian regresion on real dataset

import sys
sys.path.append("..")
from sklearn.datasets import fetch_california_housing
from utils import get_data, plot_approximation_line
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Getting data
X_train, X_test, y_train, y_test = get_data(fetch_california_housing)

# Conventional linear model
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X=X_test)

print "MSE: {:.2f}".format(mean_squared_error(y_test, y_pred))

# Regularized linear model (L1 regularization)
from sklearn.linear_model import Lasso

reg = Lasso()
reg.fit(X_train, y_train)
y_pred = reg.predict(X=X_test)

print "MSE: {:.2f}".format(mean_squared_error(y_test, y_pred))

