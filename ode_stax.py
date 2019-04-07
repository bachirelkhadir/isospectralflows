from jax.experimental.stax import glorot, randn
import jax.numpy as np
from jax import jit


def odeint(f, x0, T=1., num_steps=100, method='rk4'):
  """Integrates the function f between 0 and T starting from x0
  using Euler's or RK4 method.

  Returns:
    x(t): array of the same size as x0
  """


  # TODO: handle non-autonomous systems, i.e. the case where f
  # depends on t


  def rk4_integrator(xi, ti, dt):
    dydx = lambda t, x: f(x)
    k1 = dt*f(xi)
    k2 = dt*f(xi + k1/2.) 
    k3 = dt*f(xi + k2/2.) 
    k4 = dt*f(xi + k3)

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6.

  euler_integrator = lambda xi, ti, dt: dt * f(xi)
  integrator = rk4_integrator if method == 'rk4' else euler_integrator

  dt = T/ num_steps
  xi = x0

  for i, ti in enumerate(np.linspace(0, T, num_steps)):
    xi = xi + integrator(xi, ti, dt)

  return xi


def test_odeint():
  from scipy import integrate
  import numpy.random as npr


  d = 5
  num_steps = 1000
  tol_euler = .1
  tol_rk4 = 1e-4

  A = npr.randn(d, d)
  x0 = npr.randn(d)
  test_f = [
    lambda x: np.dot(A, x)
  ]
  T = 1.
  scipy_results = np.array([
    integrate.odeint(lambda x,t: f(x), x0, [0, T])[-1] 
    for f in test_f])
  rk4_results = np.array([
      odeint(f, x0, T=T, method='rk4', num_steps=num_steps)
    for f in test_f])
  euler_results = np.array([
      odeint(f, x0, T=T, method='euler', num_steps=num_steps)
    for f in test_f])

  assert np.linalg.norm(scipy_results - rk4_results) < tol_rk4
  assert np.linalg.norm(scipy_results - euler_results) < tol_euler



def DenseODE(W_init=glorot(), b_init=randn()):
  """Dense Ode Layer
  This layer takes as input an ndarray `x0`, runs the ode 
  `x_dot = f(x)`, where `f(x) = W2 σ(W1x + b1) + b2` and returns `x(T)`.
  """

  def init_fun(rng, input_shape):
    output_shape = input_shape
    out_dim = input_shape[-1]
    W1, b1 = W_init(rng, (input_shape[-1], out_dim)), b_init(rng, (out_dim,))
    W2, b2 = W_init(rng, (input_shape[-1], out_dim)), b_init(rng, (out_dim,))
    return output_shape, (W1, b1, W2, b2)
  
  def apply_fun(params, inputs, **kwargs):
    W1, b1, W2, b2 = params
    activation = lambda x: np.maximum(x, 0)
    f_inner = lambda x: np.dot(x, W1) + b1
    f_outer = lambda x: np.dot(x, W2) + b2
    f = lambda x: f_outer(activation(f_inner(x)))
    return odeint(f, inputs)
  
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
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (matrix_size, matrix_size)
    W1, b1 = W_init(rng, (input_shape[-1], matrix_len)), b_init(rng, (matrix_size*matrix_size,))
    W2, b2 = W_init(rng, (matrix_len, matrix_len)), b_init(rng, (matrix_size*matrix_size,))
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
    return odeint(f, H_inputs, num_steps=100)
  
  return init_fun, apply_fun

