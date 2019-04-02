import numpy as onp
import matplotlib.pyplot as plt


def fig_to_np_data(fig):
  fig.canvas.draw()
  # Now we can save it to a numpy array.
  data = onp.fromstring(fig.canvas.tostring_rgb(), dtype=onp.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data


def fig_to_data(f):
  """Takes in a function f that returns a plt.figure 
  and returns a function that returns a numpy array version of the figure
  """
  def f_np(*args, **kwargs):
    fig = f(*args, **kwargs)
    data = fig_to_np_data(fig)
    plt.close(fig)
    return data
    
  return f_np

