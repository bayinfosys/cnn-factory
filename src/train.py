import os
import sys
import logging

from os.path import join, basename

import tensorflow as tf

from tensorflow.keras import backend as K

import numpy as np

from common.loss import (dice_coef,
                        dice_coef_loss,
                        binary_cross_entropy_plus_dice_loss as bce_dice,
                        jaccard_index,
                        jaccard_index_loss)

from common.schedulers import step_decay

from generators import create_image_from_filenames_generator

from training.args import get_argument_parser, parse_arguments
from training.callbacks import create_keras_callbacks
from training.model_builder import default_model_builder, MODEL_BUILDERS


logger = logging.getLogger(__name__)

# disable loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def default_model_compiler(
        model,
        optimizer_name,
        loss_fn_name,
        metric_names,
        learning_rate
    ):
  """
  compile the model with an optimizer and loss function

  # FIXME: we can't put this in the training module because
           it needs access to the loss functions from common
  """
  import tensorflow.keras as keras

  logger.info("compiling model with: '%s', '%s', '%s', %f" % (
      optimizer_name, loss_fn_name, str(metric_names), learning_rate))

  optimizer_fns = {
      "sgd": keras.optimizers.SGD,
      "rmsprop": keras.optimizers.RMSprop,
      "adagrad": keras.optimizers.Adagrad,
      "adadelta": keras.optimizers.Adadelta,
      "adam": keras.optimizers.Adam,
      "adamax": keras.optimizers.Adamax,
      "nadam": keras.optimizers.Nadam
  }

  optimizer_args={"lr": learning_rate}

  loss_fns = {
      "bce_dice": bce_dice,
      "dice": dice_coef_loss,
      "binary_crossentropy": "binary_crossentropy",
      "jaccard": jaccard_index_loss
  }

  metric_fns = {
      "dice": dice_coef
  }

  # compile the model
  model.compile(
      optimizer=optimizer_fns[optimizer_name](**optimizer_args),
      loss=loss_fns[loss_fn_name],
      metrics=[metric_fns[m] for m in metric_names]
  )


def preprocess_image(filename):
  """
  this should be merged with the training generator
  so we replicate the image input pipeline
  """
  from skimage.transform import resize
  from skimage.io import imread

  try:
    x = imread(filename).astype(np.float)/255.0
  except ValueError as e:
    # ValueError: Could not find a format to read the specified file in single-image mode
    logger.error("%s: %s" % (filename, str(e)))
    raise FileNotFoundError(filename) from e

  x = resize(x, (256,256), anti_aliasing=True)
  return x


def preprocess_label(filename):
  from skimage.transform import resize
  from skimage.io import imread

  try:
    y = imread(filename).astype(np.uint8)
  except FileNotFoundError as e:
    logger.error("could not load: '%s'" % filename)
    raise

  y = (y==1).astype(np.float)
  y = resize(y, (256, 256), anti_aliasing=True)

  return y[..., np.newaxis]


def validate_image(X):
  if ((len(X.shape) != 3) or
      (X.shape[0] != 256) or
      (X.shape[1] != 256) or
      (X.shape[2] != 3)):
    logger.warning("[%s] expected [%s]" % (str(X.shape), str((256,256,3))))
    return False

  return True

def validate_label(X):
  if ((len(X.shape) != 3) or
      (X.shape[0] != 256) or
      (X.shape[1] != 256) or
      (X.shape[2] != 1)):
    logger.warning("[%s] expected [%s]" % (str(X.shape), str((256,256))))
    return False

  return True


if __name__ == "__main__":
  """
  train a model from the command line

  output_dir: absolute path to the base directory for writing models and training info;
              model is written to <output_dir>/<modelname>.hdf5
              tensorboard logs are under <output_dir>/logs/<modelname> (point tensorboard at <output_dir>/logs/ to compare all models)
              other training outputs are written under <output_dir>/<modelname>/

  train a simple model which just generates a mask

  modelname: name of the model which is used to create directories for storage
  model_build_fn: function which returns the model architecture to train
  model_compiler_fn: function which compiles the model with optimizer, cost functions, metrics, etc
  images: list of filenames for training data (Xs)
  masks: list of filenames for label data (ys)
  batch_size: batch size for training (fixed at 1 for the moment)
  num_augs: number of augmentations to apply to the training data (not applied to validation)
  num_epochs: maximum number of epochs for which to train the model
  learning_rate: learning rate for the model optimizer
  shuffle_data: shuffle the data each epoch if true

  TODO: bit of an abstraction issue: we called image_preprocess_fn and label_preprocess_fn
        as if they apply to images, but they are actually responsible for loading the images.
        The same pipeline could be used to train non-image data by passing different functions
        to these positions (loading sounds, text files, etc). Not sure if other abstractions
        would hold under that change.
  """
  import sys
  import os
  from os.path import exists
  from glob import glob

  log_format = "[%(asctime)s] - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  root.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

  ch = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter(log_format)
  ch.setFormatter(formatter)
  root.addHandler(ch)

  parser = get_argument_parser()
  args = parse_arguments(parser)

  images = sorted(glob(args.images))
  masks = sorted(glob(args.masks))

  logger.info("found %i/%i images/masks" % (len(images), len(masks)))
  assert len(images) > 0

  # get the intersection of the images and masks so we only get images with masks and vice-versa
  # FIXME: if we have a dir wildcard in the search (imgs/*/*.png) basename will not give the correct results
  # FIXME: if the images are a different file type, we get a mismatch
  image_basenames = set([basename(f).split(".")[0] for f in images])
  mask_basenames = set([basename(m).split(".")[0] for m in masks])

  images = [f for f, b in zip(images, image_basenames) if b in mask_basenames]
  masks  = [m for m, b in zip(masks, mask_basenames)  if b in image_basenames]

  logger.info("found %i/%i overlapping images/masks" % (len(images), len(masks)))
  # FIXME: improve error reporting here, list the directories searched, glob params, etc
  assert len(images) > 0
  assert len(images) == len(masks)


  # control memory allocation by tf
#  K.set_image_dim_ordering("tf")
  #config = tf.ConfigProto()
#  config.gpu_options.per_process_gpu_memory_fraction = 0.75
  #config.gpu_options.allow_growth = True
  #K.set_session(tf.Session(config=config))

  # MODEL
  model = default_model_builder(
              model_type=args.modeltype,
              image_shape=tuple(args.image_shape),
              channel_count=3,
              network_depth=args.network_depth
          )

  print(model.summary())

  default_model_compiler(
      model,
      optimizer_name=args.optimizer,
      loss_fn_name=args.loss_function,
      metric_names=args.training_metrics,
      learning_rate=args.learning_rate
  )

  callbacks = create_keras_callbacks(
                  model,
                  args.modelname,
                  args.output_path
              )


  # DATA
  from sklearn.model_selection import train_test_split
  train_x, test_x, train_y, test_y = train_test_split(images, masks, test_size=0.3)

  # TRAINING STEPS
  training_steps = len(train_x) // args.batch_size
  validation_steps = len(test_x) // args.batch_size

  logger.info("%i training steps, %i validation steps", training_steps, validation_steps)

  training_generator = create_image_from_filenames_generator(
      train_x,
      train_y,
      image_preprocess_fn=preprocess_image,
      label_preprocess_fn=preprocess_label,
      image_validation_fn=validate_image,
      label_validation_fn=validate_label,
      shuffle_data=args.shuffle_data,
      # augmentation_fn=default_augmentation,
      # num_args = args.num_augs
  )()

  validation_generator = create_image_from_filenames_generator(
      test_x,
      test_y,
      image_preprocess_fn=preprocess_image,
      label_preprocess_fn=preprocess_label,
      image_validation_fn=validate_image,
      label_validation_fn=validate_label,
      shuffle_data=args.shuffle_data,
  )()

  # TRAIN
  model.fit(training_generator,
            steps_per_epoch=training_steps*(1+args.num_augs),
            epochs=args.num_epochs,
            validation_data = validation_generator,
            validation_steps = validation_steps,
            #validation_data=validation_data,
            callbacks=callbacks,
            #verbose=verbosity,
            max_queue_size=1,
            workers=1,
            use_multiprocessing=False)

  # FIXME: can we ditch this? the checkpoint should handle all this for us
#  model_filename = join(output_dir, modelname + ".hdf5")
#  logger.info("saving model to '%s'" % model_filename)
#  model.save(model_filename)
