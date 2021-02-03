import os
import sys
import logging

from os.path import join, basename

import tensorflow as tf

import numpy as np

from common.loss import (dice_coef,
                        dice_coef_loss,
                        binary_cross_entropy_plus_dice_loss as bce_dice,
                        jaccard_index,
                        jaccard_index_loss)

from common.schedulers import step_decay

from generators import (create_image_from_filenames_generator,
                        create_image_augmentation_fn,
                        csv_to_lists)

from training.args import get_argument_parser, parse_arguments
from training.callbacks import create_keras_callbacks
from training.model_builder import default_model_builder, MODEL_BUILDERS

from preprocessing import (default_input_loader,
                           default_label_loader,
                           validate_tensor_shape)


logger = logging.getLogger(__name__)

# disable loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# HACK tf 4.2.1 memory leak
for gpu in tf.config.experimental.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(gpu, True)
# HACK


def default_model_compiler(
        model,
        optimizer_name,
        metric_names,
        learning_rate,
        output_definitions=None
    ):
  """
  compile the model with an optimizer and loss function

  # FIXME: we can't put this in the training module because
           it needs access to the loss functions from common
  """
  import tensorflow.keras as keras
  from tensorflow.keras.losses import CategoricalCrossentropy

  logger.info("compiling model with: optim='%s', metrics='%s', lr=%f" % (
      optimizer_name, str(metric_names), learning_rate))

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
      "jaccard": jaccard_index_loss,
      "categorical_crossentropy": "categorical_crossentropy",
  }

  metric_fns = {
      "dice": dice_coef
  }

  loss = {k:loss_fns[v["loss"]] for k,v in output_definitions.items()}
  weights = {k:v["weight"] for k,v in output_definitions.items()}

  # compile the model
  model.compile(
      optimizer=optimizer_fns[optimizer_name](**optimizer_args),
      loss=loss,
      loss_weights=weights,
      metrics=[metric_fns[m] for m in metric_names] if metric_names is not None else None,
  )


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
  import json

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

  # load the data definition from the csv or input globs
  if args.csv is not None:
    data = csv_to_lists(args.csv)
    logger.info("read %i/%i rows/columns" % (len(list(data.values())[0]), len(data.keys())))

    input_keys = args.inputs
    output_definitions = {}

    for oks in [json.loads(x) for x in args.outputs]:
      output_definitions.update(oks)

    assert input_keys is not None, "require input column names when using csvfile"
    assert output_definitions is not None, "require output column names when using csvfile"

    logger.info("inputs: '%s'" % str(input_keys))
    logger.info("outputs: '%s'" % str(output_definitions))

    # create sets of tuples for the inputs and ouputs
    inputs = list(zip(*[data[k] for k in input_keys]))
    outputs = list(zip(*[data[k] for k in output_definitions]))
  elif args.images is not None and args.labels is not None:
    # FIXME: scrap this glob loading; if people want to glob they can build a csv...
    # read the image filenames
    logger.warning("GLOB LOADING IS DEPRECATED, GO FUCK YOUSELF")
    inputs = [(x,) for x in sorted(glob(args.images))]
    outputs = [(y,) for y in sorted(glob(args.masks))]

    input_keys = None
    output_definitions = None
  else:
    raise NotImplementedError("Cannot generate data from nothing")

  logger.info("found %i/%i inputs/outputs" % (len(inputs), len(outputs)))
  assert len(inputs) > 0
  assert len(outputs) > 0
  assert len(inputs) == len(outputs)

  # FIXME: here we want to load a function from /user.py given the function name in args
  # TODO: should we have these functions accept the args variable? we could wrap all the
  #       preproc/validators in a lambda and let them specialise based on runtime args
  data_preprocess_fn = default_input_loader if args.data_preprocess_fn is None else None

#  label_preprocess_fn = default_label_loader if args.label_preprocess_fn is None else None
  logger.debug("types: '%s'" % str([o["type"] for o in output_definitions.values()]))
  logger.debug("shapes: '%s'" % str([o["size"] for o in output_definitions.values()]))
  if args.label_preprocess_fn is None:
    label_preprocess_fn = lambda y: default_label_loader(y, types=[o["type"] for o in output_definitions.values()], shapes=[o["size"] for o in output_definitions.values()])
  else:
    label_preprocess_fn = args.data_preprocess_fn


  data_validation_fn = validate_tensor_shape if args.data_validation_fn is None else None
  label_validation_fn = validate_tensor_shape if args.label_validation_fn is None else None

  # DATA
  from sklearn.model_selection import train_test_split
  train_x, test_x, train_y, test_y = train_test_split(inputs, outputs, test_size=0.3)

  # FIXME: get the name of the augmentation function from args
  augmentation_fn = create_image_augmentation_fn(args.num_augs)

  # TRAINING STEPS
  # FIXME: run through the data to remove invalid data from the counts
  training_steps = len(train_x) // args.batch_size
  validation_steps = len(test_x) // args.batch_size

  logger.info("%i training steps, %i validation steps", training_steps, validation_steps)

  training_generator = create_image_from_filenames_generator(
      train_x,
      train_y,
      image_preprocess_fn=data_preprocess_fn,
      label_preprocess_fn=label_preprocess_fn,
      image_validation_fn=lambda x: data_validation_fn(x, tuple(args.image_shape) + (3,)),
      label_validation_fn=lambda x: label_validation_fn(x, tuple(args.image_shape) + (1,)),
      shuffle_data=args.shuffle_data,
      augmentation_fn=augmentation_fn,
  )()

  validation_generator = create_image_from_filenames_generator(
      test_x,
      test_y,
      image_preprocess_fn=data_preprocess_fn,
      label_preprocess_fn=label_preprocess_fn,
      image_validation_fn=lambda x: data_validation_fn(x, tuple(args.image_shape) + (3,)),
      label_validation_fn=lambda x: label_validation_fn(x, tuple(args.image_shape) + (1,)),
      shuffle_data=args.shuffle_data
  )()

  # FIXME: run through the generators here so we use the
  #        validation functions to verify the data.

  # MODEL
  model = default_model_builder(
              model_type=args.modeltype,
              image_shape=tuple(args.image_shape),
              channel_count=3,
              network_depth=args.network_depth,
              filter_sizes=args.filter_sizes,
              output_definitions=output_definitions,
          )

  print(model.summary())

  default_model_compiler(
      model,
      optimizer_name=args.optimizer,
      metric_names=args.training_metrics,
      learning_rate=args.learning_rate,
      output_definitions=output_definitions
  )

  callbacks = create_keras_callbacks(
                  model,
                  args.modelname,
                  args.output_path
              )

  # TRAIN
  model.fit(training_generator,
            steps_per_epoch=training_steps*(1+args.num_augs),
            epochs=args.num_epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            #validation_data=validation_data,
            callbacks=callbacks,
            #verbose=verbosity,
            max_queue_size=1,
            workers=1,
            use_multiprocessing=False)
