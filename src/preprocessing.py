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


def default_label_filename_preprocess(filename, label_set=None):
  """
  default preprocessing for label image filenames is
  to load the file from disk as uint8, converted to a
  boolean array for training.

  multiple labels can be merged into one bool array
  by passing a callable or a list as label_set parameter:
  if label_set is callable: (keep if label_set(<label_id>) is true)
  if label_set is list: (keep if <label_id> in label_set)
  if label_set is int: (keep if <label_id> == label_set)

  returns bool map as a float numpy array
  """
  from skimage.io import imread

  try:
    y = imread(filename).astype(np.uint8)
  except FileNotFoundError as e:
    logger.error("could not load: '%s'" % filename)
    raise

  existing_labels = np.unique(y)
  requested_labels = []

  if callable(label_set):
    requested_labels = [l for l in existing_labels if label_set(l)]
  elif isinstance(label_set, list):
    requested_labels = [l for l in existing_labels if l in label_set]
  elif isinstance(label_set, int):
    requested_labels = [label_set]
  else:
    raise NotImplementedException("unknown label_set type: '%s'" % type(label_set))

  label_img = np.isin(y, requested_labels).astype(float)

  return label_img[..., np.newaxis]


def validate_tensor_shape(Xs, expected_shape):
  """
  return true if the shape of X matches the tuple expected_shape
  false otherwise
  """
  def check(x, ex):
    return len(x.shape) == len(ex) and all([i == j for i, j in zip(x.shape, ex)])

  if isinstance(Xs, list): # mult input
    assert isinstance(expected_shape, list), "when Xs is list, expected_shape must also be list"
    return all([check(x, ex) for x, ex in zip(Xs, expected_shape)])
  else:
    return check(Xs, expected_shape)


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


def default_label_loader(ys, types, shapes, labels=None):
  """
  load a label data from a tuple with one or more entries

  # TODO: to align the types, shapes and label definitions
  #       with the ys array we use the index of the ys iterator
  #       which means, we require types, shapes and labels to
  #       be same-shaped. Much prefer to have key-name indexing
  #       by passing the input_definitions through to this level
  """
  from os.path import exists

  i = []

  for idx, y in enumerate(ys):
    if types[idx] == "image":
      assert isinstance(y, str) and exists(y), "'%s' does not exist" % str(y)
      v = default_image_filename_preprocess(y)
    elif types[idx] == "segmentation":
      # logger.debug("loading image '%s'" % y)
      assert isinstance(y, str) and exists(y), "'%s' does not exist" % str(y)

      input_labels = None if labels is None else labels[idx]

      v = default_label_filename_preprocess(y, label_set=input_labels)
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
