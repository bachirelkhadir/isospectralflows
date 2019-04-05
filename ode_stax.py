from jax.experimental.stax import glorot, randn
import jax.numpy as np


def odeint(f, x0, T=1., num_steps=100, method='euler'):
  """Integrates the function f between 0 and T starting from x0
  using Euler's or RK4 method.

  Returns:
    x(t): array of the same size as x0
  """


  # TODO: handle non-autonomous systems, i.e. the case where f
  # depends on t


  def rk4_integrator(xi, ti, dt):
    xi = x0
    dydx = lambda t, x: f(x)
    k1 = dydx(ti, xi)
    k2 = dydx(ti + 0.5 * dt, xi + 0.5 * k1) 
    k3 = dydx(ti + 0.5 * dt, xi + 0.5 * k2) 
    k4 = dydx(ti + dt, xi + k3) 
    return dt * (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 

  euler_integrator = lambda xi, ti, dt: dt * f(xi)

  integrator = rk4_integrator if method == 'rk4' else euler_integrator

  dt = T/ num_steps
  xi = x0

  for i, ti in enumerate(np.linspace(0, T, num_steps)):
    xi = xi + integrator(xi, ti, dt)
  return xi


def test_odeint():
  from scipy import integrate
  test_f = [
    lambda x: np.array([-x[1], x[0]])
  ]

  x0 = np.array([1., 1.])
  T = 1.
  scipy_results = np.array([
    integrate.odeint(lambda x,t: f(x), x0, [0, T])[-1] 
    for f in test_f])
  rk4_results = np.array([
      odeint(f, x0, T=T, method='rk4')
    for f in test_f])
  euler_results = np.array([
      odeint(f, x0, T=T, method='euler')
    for f in test_f])
  print("scipy", scipy_results)
  print("rk4", rk4_results)
  print("euler", euler_results)
  return


def DenseODE(W_init=glorot(), b_init=randn()):
  """Dense Ode Layer
  This layer takes as input an ndarray `x0`, runs the ode 
  `x_dot = f(x)`, where `f(x) = W2 σ(W1x + b1) + b2` and returns `x(T)`.
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

