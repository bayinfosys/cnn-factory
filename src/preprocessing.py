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

  if len(x.shape) == 2 or x.shape[-1] != 3:
    x = np.stack((x,)*3, axis=-1)

  return x


def default_label_filename_preprocess(filename, min_label_value=1, max_label_value=128):
  """
  default preprocessing for label image filenames is
  to load the file from disk as uint8, convert to a
  boolean array where min_label_value <= value < max_label_value
  return that as a float numpy array
  """
  from skimage.io import imread

  try:
    y = imread(filename).astype(np.uint8)
  except FileNotFoundError as e:
    logger.error("could not load: '%s'" % filename)
    raise

  y = ((y>=min_label_value) & (y<max_label_value)).astype(np.float)

  return y[..., np.newaxis]


def validate_tensor_shape(X, expected_shape):
  """
  return true if the shape of X matches the tuple expected_shape
  false otherwise
  """
  # FIXME: I don't know how to pass this expected_shape through
  #        How can we know the expected shape from the cli
  return True
#  assert isinstance(X, type(expected_shape))
#  assert len(X) == len(expected_shape)
#
#  for x, ex in zip(X, expected_shape):
#    assert all([i == j for i, j in zip(x.shape, ex)])
#
#  return True


def convert_to_unit_range(image):
  """convert the image to a unit range
  """
  nan_idx = np.isnan(image)
  img_min = image[~nan_idx].min()
  img_max = image[~nan_idx].max()
  return (image-img_min)/(img_max-img_min)


def convert_to_zscore(image):
  """compute the mean and std without nans and without zeros as nans
     convert the image to z-score of the mean/std values
  """
  nan_idx = np.isnan(image)
  img_mu = image[~nan_idx].mean()
  img_std = image[~nan_idx].std()
  return (image-img_mu)/(img_std)


def convert_nan_to_zero(image):
  """zero out the nans in both the image and label
     FIXME: if we have a labelmap associate with this
            image we should alter the label in the same
            locations?
  """
  nan_idx = np.isnan(image)
  image[nan_idx] = 0


def default_input_loader(Xs):
  """
  load input data from a tuple with one or more entries
  """
  from os.path import exists

  i = []

  # FIXME: somehow, take a bool->fn map as parameter to invert
  #        the data type handling, so we are not tied to these rules
  for x in Xs:
    if isinstance(x, str):
      if not exists(x):
        logger.warning("'%s' does not exist" % x)
        continue
      else:
        v = default_image_filename_preprocess(x)
    else:
      v = np.array([x]).astype(np.float)

    i.append(v)

  return i


def default_label_loader(ys, types, shapes):
  """
  load a label data from a tuple with one or more entries
  """
  from os.path import exists

  i = []

  for idx, y in enumerate(ys):
    if types[idx] == "image" or types[idx] == "segmentation":
      assert isinstance(y, str) and exists(y)
      v = default_label_filename_preprocess(y)
    elif types[idx] == "category":
      # one-hot
      v = np.zeros(shapes[idx]).astype(np.float)
      v[int(y)-1] = 1.0
    elif types[idx] == "numeric":
      v = np.array([y]).astype(np.float)
    else:
      raise NotImplementedError("cannot process type of element %i (%s)" % (idx, str(ys)))

    i.append(v)

  return i
