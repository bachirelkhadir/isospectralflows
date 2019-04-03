import argparse

parser = argparse.ArgumentParser(description='Learn to classify a CIFAR-10')
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
  type=str, default="[Flatten, Dense(10),Softmax]",
  help="Use ODE layer")
parser.add_argument('--log_dir', dest='log_dir', 
  type=str, default="/tmp/checkpoints/",
  help="Directory where to store tensorboard logs.")
parser.add_argument('--log_eps', dest='log_eps', 
  type=float, default=1e-2,
  help='A constant to add the inputs of np.log to avoid log(0).')
parser.add_argument('--l2_reg', dest='l2_reg', 
  type=float, default=0.,
  help='Multiplies the l2 norm of the parameter vectors and added to the objective function.')


args = parser.parse_args()
print(args)


# params
step_size = args.step_size
num_epochs = args.num_epochs
batch_size = args.batch_size
tag = args.tag
layers_str = args.layers_str
log_dir = args.log_dir
log_eps = args.log_eps
l2_reg = args.l2_reg

# Imports
import datetime
import itertools
import jax
import jax.numpy as np
import keras
import matplotlib.pyplot as plt
import numpy as onp
import numpy.random as npr
import os
import pickle
import tarfile
import tensorboard_logging
import time
import zipfile 

from ode_stax import *

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
from plot_help import *




# init randomness
key = jax.random.PRNGKey(0)
npr.seed(0)


# Download datasets
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 250.).astype(np.float64).reshape(-1, 28, 28, 1)
x_test = (x_test / 250.).astype(np.float64).reshape(-1, 28, 28, 1)
num_classes = 10
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# Build NN
@jit
def vec(params):
  """Stack parameters in a big vector."""
  
  leaves, _ = tree_flatten(params)
  return np.hstack([x.flatten() for x in leaves])


def cross_entropy(a, b):
  return - np.sum(a * np.log(b+log_eps))


@jit
def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs, rng=key)
  return cross_entropy(targets, preds) + l2_reg * np.linalg.norm(vec(params))


@jit
def accuracy(params, batch):
  inputs, targets = batch
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(predict(params, inputs, rng=key), axis=1)
  return np.mean(predicted_class == target_class)


print("Evaluating layers from string:", layers_str)
layers = eval(layers_str)
init_random_params, predict = stax.serial(*layers)



    
################
# Optimizer
################

# start adam optimizer and initialize parameters
opt_init, opt_update = optimizers.adam(step_size)
out_shape, init_params = init_random_params((-1,) + x_train.shape[1:])
opt_state = opt_init(init_params)
n_params = len(vec(init_params))

# helper function to perform a one step update of the parameters
@jit
def update(i, opt_state, batch):
  params = optimizers.get_params(opt_state)
  return opt_update(i, grad(loss)(params, batch), opt_state)


print("Gettting batches")
data_streamer = DataStreamer(x_train, y_train, batch_size=batch_size, num_classes=num_classes)
data_streamer_test = DataStreamer(x_test, y_test, batch_size=batch_size, num_classes=num_classes)

itercount = itertools.count()
print("Starting logger")
logger = tensorboard_logging.create_logger(tag=tag.format(n_params=n_params), log_dir=log_dir)
print("Number of params %.4fk" % (n_params/1e3))

# Start iterating
print("|".join(map(lambda s: s.center(10, ' '), "Epoch,Loss,Accuracy,dt".split(","))))
for epoch in range(num_epochs):
    start_time = time.time()

    # loss and accuracy for each minibatch
    cum_loss = []
    cum_acc = []

    # runs through all the data
    for _ in range(data_streamer.num_batches):
      batch = next(data_streamer.stream_iter)
      i = next(itercount)
      opt_state = update(i, opt_state, batch)
      params = optimizers.get_params(opt_state)
      
      # logger
      loss_i = loss(params, batch)
      # evaluate accuracy on test batch
      batch_test = next(data_streamer.stream_iter)
      acc_i = accuracy(params, batch_test)
      cum_loss.append(loss_i)
      cum_acc.append(acc_i)
      logger.log_scalar('accuracy', acc_i, step=i+1)
      logger.log_scalar('loss', loss_i, step=i+1)

    epoch_time = time.time() - start_time
    print("{:10}|{:10.5f}|{:10.5f}|{:10.5f}".format(epoch, onp.mean(cum_loss),
                                                  onp.mean(cum_acc),
                                                  epoch_time))  
