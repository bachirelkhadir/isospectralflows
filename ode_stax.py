from jax.experimental.stax import glorot, randn
import jax.numpy as np


def euler_odeint(f, x0, T=1., num_steps=100):
  """Integrates the function f between 0 and T starting from x0
  using Euler's method

  Returns:
    x(t): array of the same size as x0
  """

  dt = T/ num_steps
  xi = x0
  for i in range(num_steps):
    xi = xi + f(xi)*dt
  return xi


def DenseODE(W_init=glorot(), b_init=randn()):
  """Dense Ode Layer
  This layer takes as input an ndarray `x0`, runs the ode 
  `x_dot = x`, where `f(x) = W2 σ(W1x + b1) + b2` and returns `x(T)`.
  """

  def init_fun(input_shape):
    output_shape = input_shape
    out_dim = input_shape[-1]
    W1, b1 = W_init((input_shape[-1], out_dim)), b_init((out_dim,))
    W2, b2 = W_init((input_shape[-1], out_dim)), b_init((out_dim,))
    return output_shape, (W1, b1, W2, b2)
  
  def apply_fun(params, inputs, **kwargs):
    W1, b1, W2, b2 = params
    activation = lambda x: np.maximum(x, 0)
    f_inner = lambda x: np.dot(x, W1) + b1
    f_outer = lambda x: np.dot(x, W2) + b2
    f = lambda x: f_outer(activation(f_inner(x)))
    return euler_odeint(f, inputs)
  
  return init_fun, apply_fun


def IsospectralODE(matrix_size, activation=None, W_init=glorot(), b_init=randn()):
  """Isospectral ODE Layer.
  
  - Maps the input `x` to a `matrix_size x matrix_size` matrix `H` using a fully connected layer.
  - Run an isospectral flow (i.e. `H_dot =  H = [[H, N], H]`) starting from `H`
  - Returns H(T).
  
  In other words,  H0 = W2 σ(W1x + b1) + b2 and H_dot = [[H, N], H].
  """

  matrix_len = matrix_size*matrix_size
  if activation == None:
    activation = lambda x: np.maximum(x, 0)
  def init_fun(input_shape):
    output_shape = input_shape[:-1] + (matrix_size, matrix_size)
    W1, b1 = W_init((input_shape[-1], matrix_len)), b_init((matrix_size*matrix_size,))
    W2, b2 = W_init((matrix_len, matrix_len)), b_init((matrix_size*matrix_size,))
    #N = W_init((matrix_size, matrix_size))
    N = np.diag(np.linspace(0, 1, matrix_size))
    return output_shape, (W1, b1, W2, b2, N,)
  
  def apply_fun(params, inputs, **kwargs):
    W1, b1, W2, b2, N = params
    N = (N + N.T)/2
    # N = np.diag(np.linspace(0, 1, matrix_size))
    activation = lambda x: np.maximum(x, 0)
    f_inner = lambda x: np.dot(x, W1) + b1
    f_outer = lambda x: (np.dot(x, W2) + b2).reshape((-1, matrix_size, matrix_size))
    H0 = lambda x: sym(f_outer(activation(f_inner(x))))
    sym = lambda M:  (M+M.transpose((0, 2, 1)))/2
    H_inputs = H0(inputs)
    
    bracket = lambda A, B: np.matmul(A, B) - np.matmul(B, A) 
    flow = lambda H: bracket(H, bracket(H, N))
    def f(H):
      return flow(H)
    return euler_odeint(f, H_inputs, num_steps=100)
  
  return init_fun, apply_fun

