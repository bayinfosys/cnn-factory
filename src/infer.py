"""
segment an image using a model
"""
import sys
import logging
import os

from os.path import join, basename

import numpy as np

import tensorflow as tf
from keras.models import load_model

from common.loss import (dice_coef,
                       dice_coef_loss,
                       binary_cross_entropy_plus_dice_loss as bce_dice,
                       jaccard_index,
                       jaccard_index_loss)

# disable loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", logging.DEBUG))

ch = logging.StreamHandler(sys.stdout)
#ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def process_image(filename):
  """
  this should be merged with the training generator
  so we replicate the image input pipeline
  """
  from skimage.transform import resize
  from skimage.io import imread
#  x = imread(filename, as_gray=True).astype(np.float) / 255.0
  x = imread(filename).astype(np.float) / 255.0
  x = resize(x, (256,256), anti_aliasing=True)
  return x[np.newaxis, ...]


if __name__ == "__main__":
  import argparse
  import json
  import sys
  import os
  from os.path import exists
  from glob import glob
  from skimage.io import imsave

  parser = argparse.ArgumentParser(
      description="infer network segmentation of image"
  )
  parser.add_argument("--model-filename", "-m", required=True, help="name of the model file")
  parser.add_argument("--output-path", default="preds", help="output directory for results")
  parser.add_argument("--images", required=True, help="file pattern of input images")
  args = parser.parse_args()

  # load the model
  model_name = basename(args.model_filename.split(".")[0])

  # metrics
  # FIXME: might need these for loading the model
  custom_loss_fns = {"dice_coef_loss": dice_coef_loss,
                     "dice_coef": dice_coef,
                     "bce+dice": bce_dice}

  # load the model
  with tf.device("/cpu:0"):
    model = load_model(args.model_filename,
                       custom_objects=custom_loss_fns,
                       compile=False)

  if model is None:
    logger.error("Could not load '%s' as keras model", args.model_filename)
    sys.exit(-1)
  else:
    logger.info("Loaded keras model '%s'", args.model_filename)

  images = sorted(glob(args.images))
  masks = [] * len(images)

  for index, image_filename in enumerate(images):
    logger.info("processing: '%s'" % image_filename)
    output_filename = join(args.output_path, basename(image_filename))

    input = process_image(image_filename)

    logger.info("predicting...")
    prediction = model.predict(input, batch_size=1, verbose=0)[0,...]
    logger.info("preds: %f -> %f" % (prediction.min(), prediction.max()))

    mask = (prediction > 0.5).astype(np.uint8)

    labels, counts = np.unique(mask, return_counts=True)
    logger.info("labels: '%s'" % (["%i: [%i]" % (l, c) for l, c in zip(labels, counts)]))
    logger.info("writing to: '%s'", output_filename)

    out_image = np.hstack([input[0, ...], np.dstack([prediction]*3), np.dstack([mask]*3)])
    imsave(output_filename, out_image*255.0)
