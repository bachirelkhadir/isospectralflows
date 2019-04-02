"""Simple example on how to log scalars and images to tensorboard without tensor ops.

License: Copyleft
"""
__author__ = "Michael Gygli"

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import scipy.misc
import tensorflow as tf
import tensorflow as tf

from io import BytesIO


LOG_DIR = '/tmp/checkpoints/'

def create_logger(tag='', log_dir=LOG_DIR):
  tmp_name = os.path.join(LOG_DIR, tag+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  print("Writing to", tmp_name)
  os.mkdir(tmp_name)    
  logger = Logger(tmp_name)
  return logger
  
  
class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def save_images_from_event(fn, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)
    list_image_paths = []
    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
              #print(v.tag)
              if v.tag == tag:
                  im = im_tf.eval({image_str: v.image.encoded_image_string})
                  output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                  list_image_paths.append(output_fn)
                  # print("Saving '{}'".format(output_fn))
                  scipy.misc.imsave(output_fn, im)
                  count += 1  
    return list_image_paths

# Test logger
# import tensorboard_logging
# logger = tensorboard_logging.create_logger('test_logger2')
# for i in range(100):
#   logger.log_scalar("mytag", -100, i)
