import argparse

parser = argparse.ArgumentParser(description='GANS')
parser.add_argument('--step_size', dest='step_size', 
  type=float, default=1e-2,
  help='Step size')
parser.add_argument('--num_epochs', dest='num_epochs', 
  type=int, default=100,
  help='Number of epochs')
parser.add_argument('--batch_size', dest='batch_size', 
  type=int, default=100,
  help='Mini-batch size')
parser.add_argument('--noise_dim', dest='noise_dim', 
  type=int, default=2,
  help='Dimension of the noise fed to the generator.')
parser.add_argument('--tag', dest='tag', 
  type=str, default="no_tag_gan_D{n_params_D}_G{n_params_G}@",
  help='Tag for logging with tensorboard')

parser.add_argument('--layers_D', dest='layers_D', 
  type=str, default="[Flatten, Dense(10),Softmax]",
  help="Layers for discriminator neural net.")
parser.add_argument('--layers_G', dest='layers_G', 
  type=str, default="[Dense(10), Relu,Dense(2)]",
  help="Layers for generator neural net.")


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
noise_dim = args.noise_dim
tag = args.tag
layers_D_str = args.layers_D
layers_G_str = args.layers_G

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
from plot_help import *


key = jax.random.PRNGKey(0)


################
# Data
################

N = 10
M = int(5e4)
theta = npr.randint(0, N, size=(M, 1)) * 2*np.pi/N
noise = npr.randn(M, 2) / 50
x_train = np.hstack([np.cos(theta), np.sin(theta)]) + noise
input_shape = (-1,)+x_train.shape[1:]
print("x shape", x_train.shape)

class NoiseStreamer:
  def __init__(self, size=100, batch_size=32):
      rng = npr.RandomState(0)

      def data_stream():
        while True:
          yield npr.randn(batch_size, size)      

      self.stream_iter = data_stream()   



##################
# NN helpers
##################


@jit
def vec(params):
  """Stack parameters in a big vector"""
  
  leaves, _ = tree_flatten(params)
  return np.hstack([x.flatten() for x in leaves])
 

@jit
def loss_G(params_G, params_D, noise_batch):
  fake_img_batch = generator(params_G, noise_batch, rng=key)\
                                          .reshape(input_shape)
  p_fake = discriminator(params_D, fake_img_batch, rng=key)
  eps = 1e-6
  return -np.mean(np.log(p_fake+eps))


@jit
def loss_D(params_D, params_G, img_batch, noise_batch):
  fake_img_batch = generator(params_G, noise_batch, rng=key)\
                                          .reshape(input_shape)

  p_real = discriminator(params_D, img_batch, rng=key)
  p_fake = discriminator(params_D, fake_img_batch, rng=key)
  eps = 1e-6
  return -np.mean(np.log(1-p_fake+eps) + np.log(p_real+eps))


Sigmoid = stax._elemwise_no_params(lambda x: 0.5 * (np.tanh(x / 2.) + 1))

D_layers = eval(layers_D_str)


G_layers = eval(layers_G_str)

init_D, discriminator = stax.serial(*D_layers)
init_G, generator = stax.serial(*G_layers)


####################################
# Plot help
####################################

import plot_help

def plot_output_G(params_G, z):
  fig = plt.figure()
  Gz = generator(params_G, z, rng=key, mode='test')
  plt.scatter(*Gz.T)                     
  plt.tight_layout()
  return fig


def plot_output_G_to_data(params_G, z):
  fig = plot_output_G(params_G, z)
  data = plot_help.fig_to_np_data(fig)
  plt.close(fig)
  return data

z = npr.randn(100, noise_dim) 


################
# Optimizer
################


step_size = 2e-4
num_epochs = 100



opt_G_init, opt_G_update = optimizers.adam(step_size)
opt_D_init, opt_D_update = optimizers.adam(step_size)

grad_G = grad(loss_G)
grad_D = grad(loss_D)


shape_G, params_G_initial = init_G((-1, noise_dim))
shape_D, params_D_initial = init_D(input_shape)

opt_G_state = opt_G_init(params_G_initial)
opt_D_state = opt_D_init(params_D_initial)

n_params_G = len(vec(params_G_initial))
n_params_D = len(vec(params_D_initial))
logger = tensorboard_logging.create_logger(tag=tag.format(n_params_D=n_params_D, n_params_G=n_params_G), 
											log_dir=log_dir)


def update_G(i, opt_G_state, opt_D_state, noise_batch):
    params_D = optimizers.get_params(opt_D_state)
    params_G = optimizers.get_params(opt_G_state)
    opt_G_state = opt_G_update(i, grad_G(params_G, params_D, noise_batch), 
                             opt_G_state)
    return opt_G_state

  
def update_D (i, opt_G_state, opt_D_state, real_batch, noise_batch):
  params_D = optimizers.get_params(opt_D_state)
  params_G = optimizers.get_params(opt_G_state)
  opt_D_state = opt_D_update(i, grad_D(params_D, params_G, real_batch, noise_batch),
                           opt_D_state)
  
  return opt_D_state



def update(i, opt_G_state, opt_D_state, real_samples_iter, noise_samples_iter):
  real_batch = next(real_samples_iter)
  noise_batch = next(noise_samples_iter)
  noise_batch = noise_batch[:len(real_batch)]  
  opt_D_state = update_D(i, opt_G_state, opt_D_state, real_batch, noise_batch)
  

  noise_batch = next(noise_samples_iter)
  opt_G_state = update_G(i, opt_G_state, opt_D_state, noise_batch)
  
  return opt_G_state, opt_D_state


itercount = itertools.count()
real_samples = DataStreamer(x_train, batch_size=batch_size)
noise_samples = NoiseStreamer(batch_size=batch_size, size=noise_dim)


print("Starting training...")
print("Number of params %.4fM + %.4fM " % 
      (n_params_D/1e6, n_params_G/1e6))

print("|".join(map(lambda s: s.center(10, ' '), "Iter,Loss D,Loss G".split(","))))


for epoch in range(num_epochs):
  start_time = time.time()
  
  for _ in range(real_samples.num_batches):
    
    # step
    i = next(itercount)
    opt_G_state, opt_D_state = update(i, opt_G_state, opt_D_state, 
                                      real_samples.stream_iter, noise_samples.stream_iter)
    
    # log
    if i % 50 == 0:
      params_G = optimizers.get_params(opt_G_state)
      params_D = optimizers.get_params(opt_D_state)
      
      real_batch = next(real_samples.stream_iter)
      noise_batch = next(noise_samples.stream_iter)
      noise_batch = noise_batch[:len(real_batch)]  
      
      loss_D_i = loss_D(params_D, params_G, real_batch, noise_batch)
      loss_G_i = loss_G(params_G, params_D, noise_batch)

      logger.log_scalar('D', loss_D_i, step=i+1)
      logger.log_scalar('G', loss_G_i, step=i+1)

      logger.log_images('G(z)', [plot_output_G_to_data(params_G, z)], step=i+1)
      print("{:10}|{:10.5f}|{:10.5f}".format(i, loss_D_i, loss_G_i))
    

  epoch_time = time.time() - start_time
  print("Dt =", epoch_time)
  





