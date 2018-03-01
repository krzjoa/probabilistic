# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Making toy dataset
def build_toy_dataset(N, w, noise_std=0.1):
  D = len(w)
  x = np.random.randn(N, D)
  y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
  return x, y

N = 40
D = 10

w_true = np.random.randn(D)
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)

# Building Edward model
import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import edward as ed


# Model
X = tf.placeholder(tf.float32, [N, D])

w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
w = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))


# Inference