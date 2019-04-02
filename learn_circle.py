# Imports

import numpy as np
import pickle
import os
import urllib
import zipfile 
import tarfile
import time
import datetime
from six.moves import urllib

# import for showing the confusion matrix
import itertools
import matplotlib.pyplot as plt

import numpy.random as npr
import jax
import jax.numpy as np
from jax import device_put
import numpy as onp

import keras
from keras.datasets import cifar10
from keras.datasets import mnist


from jax.tree_util import tree_flatten
from jax.experimental import stax
from jax.config import config
from jax import jit, grad
from jax.experimental import optimizers

from jax.experimental.stax import Dense, Relu, Softmax, Conv, \
                          Dropout, MaxPool, BatchNorm, Flatten
import ode_stax
from plot_help import *


key = jax.random.PRNGKey(0)


import tensorboard_logging


# Download datasets
print("Downloading data")
x_train = npr.randn(500, 2)
y_train = (np.linalg.norm(x_train, axis=1) < 1).astype(float)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'x train samples')
print(y_train.shape[0], 'y train samples')

class DataStreamer:
  def __init__(self, x_train, y_train, batch_size=128):
      rng = npr.RandomState(0)

      num_train = len(x_train)
      num_complete_batches, leftover = divmod(num_train, batch_size)
      num_batches = num_complete_batches + bool(leftover)
      num_classes = 2
      Id = onp.eye(num_classes)
      def data_stream():
        while True:
          perm = rng.permutation(num_train)
          for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            batch_x = x_train[batch_idx]
            batch_y = Id[y_train[batch_idx].astype(int), :]
            yield batch_x, batch_y
      
      self.num_train = num_train
      self.num_batches = num_batches
      self.stream_iter = data_stream()

# Build NN
@jit
def vec(params):
  """Stack parameters in a big vector"""
  
  leaves, _ = tree_flatten(params)
  return np.hstack([x.flatten() for x in leaves])

#@jit
def cross_entropy(a, b):  
  eps = 1e-2
  return - np.sum(a * np.log(b+eps))

@jit
def loss(params, batch, reg=0.):
  inputs, targets = batch
  preds = predict(params, inputs, rng=key)

  return cross_entropy(targets, preds) + reg * np.linalg.norm(vec(params))

@jit
def accuracy(params, batch):
  inputs, targets = batch
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(predict(params, inputs, rng=key), axis=1)
  return np.mean(predicted_class == target_class)



layers = [
    ode_stax.IsospectralODE(4), Flatten,
    Dense(2), Softmax]

init_random_params, predict = stax.serial(*layers)


# Plotting helpers
def plot_output_layer(params):
  nx, ny = (30, 30)
  X = np.linspace(-3, 3, nx)
  Y = np.linspace(-3, 3, ny)
  X, Y = np.meshgrid(X, Y)
  XY = np.array([X.flatten(), Y.flatten()]).T
  Z = predict(params, XY)
  Z = Z.reshape((nx, ny, -1))[:, :, 0]                
  
  fig = plt.figure()
  plt.contourf(X, Y, Z)
  plt.colorbar()  
  plt.tight_layout()
  data = fig_to_np_data(fig)
  plt.close(fig)
  return data


def plot_N(params):             
  fig = plt.figure()
  N = params[0][-1]
  plt.imshow(N)
  data = fig_to_np_data(fig)
  plt.close(fig)
  return data


    
################
# Optimizer
################
step_size = 1e-2
num_epochs = 100


opt_init, opt_update = optimizers.adam(step_size)

@jit
def update(i, opt_state, batch):
  params = optimizers.get_params(opt_state)
  return opt_update(i, grad(loss)(params, batch), opt_state)

out_shape, init_params = init_random_params((-1,) + x_train.shape[1:])
opt_state = opt_init(init_params)
n_params = len(vec(init_params))




print("Gettting batches")
data_streamer = DataStreamer(x_train, y_train, batch_size=100)
itercount = itertools.count()

print("Starting logger")
logger = tensorboard_logging.create_logger(tag="iso_{}_params@".format(n_params))



print("Starting training...")
print("Number of params %.4fk" % (n_params/1e3))
print("   Epoch  |  Loss  | Accuracy |      dt       ")


for epoch in range(num_epochs):
  start_time = time.time()
  cum_loss = []
  cum_acc = []
  for _ in range(data_streamer.num_batches):
    batch = next(data_streamer.stream_iter)
    i = next(itercount)
    opt_state = update(i, opt_state, batch)
    params = optimizers.get_params(opt_state)
    
    if True:
      # logger
      loss_i = loss(params, batch)
      acc_i = accuracy(params, batch)
      cum_loss.append(loss_i)
      cum_acc.append(acc_i)
      logger.log_scalar('accuracy', acc_i, step=i+1)
      logger.log_scalar('loss', loss_i, step=i+1)
      imgs = [
          plot_output_layer(params),
          plot_N(params)
      ]
      logger.log_images('ouput', imgs, step=i+1)
    
    
  epoch_time = time.time() - start_time

  
  train_loss = cum_loss
  #test_acc = accuracy(params, (x_test, y_test))
  print("{:10}|{:8.5f}|{:9.5f}|{:10.5f}".format(epoch, onp.mean(cum_loss),
                                                onp.mean(cum_acc),
                                                epoch_time))
