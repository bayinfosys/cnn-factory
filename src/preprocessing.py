"""
default preprocessing functions for input data

These functions are passed to train.py and used by generators to load and validate data
passed to the training process.

These functions should be provided by calling code to ensure data is correctly
loaded and validated for particular data use-cases.

image filename data is loaded by:
+ default_image_filename_preprocess
+ default_label_filename_preprocess

image shape validation is provided by:
+ validate_image_shape
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def default_image_filename_preprocess(filename):
  """
  default preprocessing for image filenames is simply
  to load the file from disk as a float, convert to 0..1
  and return the numpy array
  """
  from skimage.io import imread

  try:
    x = imread(filename).astype(np.float)/255.0
  except ValueError as e:
    # ValueError: Could not find a format to read the specified file in single-image mode
    logger.error("%s: %s" % (filename, str(e)))
    raise FileNotFoundError(filename) from e

  return x


def default_label_filename_preprocess(filename, label_value=1):
  """
  default preprocessing for label image filenames is
  to load the file from disk as uint8, convert to a
  boolean array where values == label_value and
  return that as a float numpy array
  """
  from skimage.io import imread

  try:
    y = imread(filename).astype(np.uint8)
  except FileNotFoundError as e:
    logger.error("could not load: '%s'" % filename)
    raise

  y = (y==label_value).astype(np.float)

  return y[..., np.newaxis]


def validate_image_shape(X, input_shape):
  """
  return true if the shape of X matches the tuple input_shape
  false otherwise
  """
  if ((len(X.shape) != len(input_shape)) or
      (not all([i == j for i, j in zip(X.shape, input_shape)]))):
    logger.warning("[%s] expected [%s]" % (str(X.shape), str(input_shape)))
    return False

  return True
