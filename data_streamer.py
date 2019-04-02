import numpy as onp
import numpy.random as npr


class DataStreamer:
  def __init__(self, x_train, y_train=None, batch_size=128, num_classes=None):
      rng = npr.RandomState(0)

      num_train = len(x_train)
      num_complete_batches, leftover = divmod(num_train, batch_size)
      num_batches = num_complete_batches + bool(leftover)

      if num_classes:
        Id = onp.eye(num_classes)

      def data_stream():
        while True:
          perm = rng.permutation(num_train)
          for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            batch_x = x_train[batch_idx]
            if y_train is None:
              batch_y = y_train[batch_idx]
              if num_classes:
                batch_y = Id[y_train[batch_idx].astype(int), :]
              yield batch_x, batch_y
            else:
              yield batch_x
      
      self.num_train = num_train
      self.num_batches = num_batches
      self.stream_iter = data_stream()

      