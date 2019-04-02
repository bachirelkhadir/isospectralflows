import numpy as onp

def fig_to_np_data(fig):
  fig.canvas.draw()
  # Now we can save it to a numpy array.
  data = onp.fromstring(fig.canvas.tostring_rgb(), dtype=onp.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data
