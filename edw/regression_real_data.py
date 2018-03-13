# Bayesian regresion on real dataset

import sys
sys.path.append("..")
from sklearn.datasets import fetch_california_housing
from utils import get_data, plot_approximation_line
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Getting data
X_train, X_test, y_train, y_test = get_data(fetch_california_housing)

# Data size
N, D  = X_train.shape
Nt, _ = X_test.shape


# Conventional linear model
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X=X_test)

print "Linear Regression MSE: {:.2f}".format(mean_squared_error(y_test, y_pred))

# Regularized linear model (L1 regularization)
from sklearn.linear_model import Lasso

reg = Lasso()
reg.fit(X_train, y_train)
y_pred = reg.predict(X=X_test)

print "Lasso MSE: {:.2f}".format(mean_squared_error(y_test, y_pred))

## Bayesian Linear regression
import tensorflow as tf
import edward as ed
from edward.models import Normal

x = tf.placeholder(tf.float32, [None, D])

w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(x, w) + b, scale=tf.ones(N))
yt = Normal(loc=ed.dot(x, w) + b, scale=tf.ones(Nt))

qw = Normal(loc=tf.get_variable("qw.loc", [D]),
            scale=tf.nn.softplus(tf.get_variable("qw.scale", [D])))
qb = Normal(loc=tf.get_variable("qb.loc", [1]),
            scale=tf.nn.softplus(tf.get_variable("qb.scale", [1])))

inference = ed.KLqp({w: qw, b: qb}, data={x: X_train, y: y_train})
inference.run(n_samples=2000, n_iter=1000)

y_post = ed.copy(yt, {w: qw, b: qb})
print("Mean squared error on test data:")
# print(ed.evaluate('mean_squared_error', data={x: X_test, y_post: y_test}, n_samples=1000))
# print(ed.evaluate('mean_squared_error', data={x: X_test, y_post: y_test}))
print(ed.evaluate('mean_squared_error', data={x: X_test, y_post: y_test}, n_samples=10))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={x: X_test, y_post: y_test}))

