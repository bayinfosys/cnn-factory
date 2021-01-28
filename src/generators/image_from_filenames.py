import logging
import numpy as np

from os.path import basename
from sklearn.utils import shuffle


logger = logging.getLogger(__name__)


def create_image_from_filenames_generator(
        Xs,
        ys,
        image_preprocess_fn=None,
        label_preprocess_fn=None,
        image_validation_fn=None,
        label_validation_fn=None,
        augmentation_fn=None,
        shuffle_data=False,
        debug_output_path=None):
  """
  return a generator to yield images from filenames

  Xs: list of image filenames
  ys: aligned list of outputs
  image_preprocess_fn: preprocessing function to x in Xs, f(str) -> np.matrix, raise FileNotFoundError on error
  label_preprocess_fn: preprocessing function to y in ys, f(str) -> np.matrix, raise FileNotFoundError on error
  image_validation_fn: return true to continue processing the image, false otherwise
  label_validation_fn: return true to continue processing the label, false otherwise
  augumentation_fn: function which returns new instances of data passed to it (also validated by generator)
  shuffle_data: shuffle the input lists on each epoch if true
  debug_output_path: absolute path to write the yielded images for debugging
  """
  if image_preprocess_fn is None:
    image_preprocess_fn = lambda x: x

  if label_preprocess_fn is None:
    label_preprocess_fn = lambda x: x

  if image_validation_fn is None:
    image_validation_fn = lambda x: True

  if label_validation_fn is None:
    label_validation_fn = lambda x: True

  def gen():
    epoch_idx = 0

    while True:
      if shuffle_data:
        Xs_, ys_ = shuffle(Xs, ys)
      else:
        Xs_, ys_ = Xs, ys

      epoch_idx = epoch_idx + 1

      for X, y in zip(Xs_, ys_):
        try:
          i_x = image_preprocess_fn(X)
          i_y = label_preprocess_fn(y)
        except FileNotFoundError as e:
          logger.error("Could not load '%s' [%s]" % (X, y))
          Xs.remove(X)
          ys.remove(y)
          continue
        except Exception as e:
          logger.exception(e)
          continue

        if image_validation_fn(i_x) is False:
          logger.warning("'%s' failed image validation" % X)
          Xs.remove(X)
          ys.remove(y)
          continue

        if label_validation_fn(i_y) is False:
          logger.warning("'%s' failed label validation" % y)
          Xs.remove(X)
          ys.remove(y)
          continue

        if debug_output_path is not None:
          write_label_image(debug_output_path, epoch_idx, basename(X), i_x, i_y)

        # logger.debug("yielding: '%s'/'%s'" % (str(i_x.shape), str(i_y.shape)))
        # NB: this is the incantation which turns the generator data into
        # batch compatible shapes (just adds a new axis at the front).
        yield [np.array([x]) for x in i_x], [np.array([y]) for y in i_y]

        if augmentation_fn is not None:
          for aug_id, (aug_x, aug_y) in enumerate(augmentation_fn(i_x, i_y)):
            if debug_output_path is not None:
              write_label_image(debug_output_path, epoch_idx,
                  "%s-aug-%04i" % (basename(aug_x), aug_id), aug_x, aug_y)

            if image_validation_fn(i_x) is False:
              logger.warning("'%s' failed image validation" % X)
              continue

            if label_validation_fn(i_y) is False:
              logger.warning("'%s' failed label validation" % y)
              continue

            # logger.debug("yielding: '%s'/'%s'" % (str(i_x.shape), str(i_y.shape)))
            yield [np.array([x]) for x in aug_x], [np.array([y]) for y in aug_y]


  return gen
