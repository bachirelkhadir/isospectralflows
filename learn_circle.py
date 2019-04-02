import argparse

parser = argparse.ArgumentParser(description='Learn to classify a circle')
parser.add_argument('--step_size', dest='step_size', 
  type=float, default=1e-2,
  help='Step size')
parser.add_argument('--num_epochs', dest='num_epochs', 
  type=int, default=100,
  help='Number of epochs')
parser.add_argument('--batch_size', dest='batch_size', 
  type=int, default=100,
  help='Mini-batch size')
parser.add_argument('--tag', dest='tag', 
  type=str, default="no_tag_{n_params}_params@",
  help='Tag for logging with tensorboard')

parser.add_argument('--layers', dest='layers_str', 
  type=str, default="[Dense(2),Softmax]",
  help="Use ODE layer")


args = parser.parse_args()
print(args)


# params
step_size = args.step_size
num_epochs = args.num_epochs
batch_size = args.batch_size
tag = args.tag
layers_str = args.layers_str

# Imports
import datetime
import itertools
import jax
import jax.numpy as np
import keras
import matplotlib.pyplot as plt
import numpy as onp
import numpy.random as npr
from ode_stax import *
import os
import pickle
import tarfile
import tensorboard_logging
import time
import zipfile 

from data_streamer import DataStreamer
from jax import device_put
from jax import grad
from jax import jit
from jax.config import config
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import BatchNorm
from jax.experimental.stax import Conv
from jax.experimental.stax import Dense
from jax.experimental.stax import Dropout
from jax.experimental.stax import Flatten
from jax.experimental.stax import MaxPool
from jax.experimental.stax import Relu
from jax.experimental.stax import Softmax
from jax.tree_util import tree_flatten
from keras.datasets import cifar10
from keras.datasets import mnist
from plot_help import *




# init randomness
key = jax.random.PRNGKey(0)
npr.seed(0)


# Download datasets
print("Downloading data")
x_train = npr.randn(5000, 2)
y_train = (np.linalg.norm(x_train, axis=1) < 1).astype(float)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'x train samples')
print(y_train.shape[0], 'y train samples')


# Build NN
@jit
def vec(params):
  """Stack parameters in a big vector"""
  
  leaves, _ = tree_flatten(params)
  return np.hstack([x.flatten() for x in leaves])


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


print("Evaluating layers from string")
layers = eval(layers_str)

init_random_params, predict = stax.serial(*layers)


# Plotting helpers

@fig_to_data
def plot_output_layer(params, predict_fun):
  nx, ny = (30, 30)
  X = np.linspace(-3, 3, nx)
  Y = np.linspace(-3, 3, ny)
  X, Y = np.meshgrid(X, Y)
  XY = np.array([X.flatten(), Y.flatten()]).T
  Z = predict_fun(params, XY)
  Z = Z.reshape((nx, ny, -1))[:, :, 0]                
  
  fig = plt.figure()
  plt.contourf(X, Y, Z)
  plt.colorbar()  
  plt.tight_layout()
  return fig


@fig_to_data
def plot_N(params):             
  fig = plt.figure()
  N = params[0][-1]
  plt.imshow(N)
  return fig

    
################
# Optimizer
################


opt_init, opt_update = optimizers.adam(step_size)

@jit
def update(i, opt_state, batch):
  params = optimizers.get_params(opt_state)
  return opt_update(i, grad(loss)(params, batch), opt_state)

out_shape, init_params = init_random_params((-1,) + x_train.shape[1:])
opt_state = opt_init(init_params)
n_params = len(vec(init_params))


print("Gettting batches")
data_streamer = DataStreamer(x_train, y_train, batch_size=batch_size, num_classes=2)
itercount = itertools.count()

print("Starting logger")
logger = tensorboard_logging.create_logger(tag=tag.format(n_params=n_params))


print("Starting training...")
print("Number of params %.4fk" % (n_params/1e3))

print("|".join(map(lambda s: s.center(10, ' '), "Epoch,Loss,Accuracy,dt".split(","))))


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
            plot_output_layer(params, predict),
            #plot_N(params)
        ]
        logger.log_images('ouput', imgs, step=i+1)
      
      
    epoch_time = time.time() - start_time

    
    train_loss = cum_loss
    #test_acc = accuracy(params, (x_test, y_test))
    print("{:10}|{:10.5f}|{:10.5f}|{:10.5f}".format(epoch, onp.mean(cum_loss),
                                                  onp.mean(cum_acc),
                                                  epoch_time))  
