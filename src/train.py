import sys
import logging

from os import makedirs
from os.path import join, basename

from keras import backend as K

#from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np

from common.loss import (dice_coef,
                        dice_coef_loss,
                        binary_cross_entropy_plus_dice_loss as bce_dice,
                        jaccard_index,
                        jaccard_index_loss)

from common.schedulers import step_decay


logger = logging.getLogger(__name__)

# disable loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def model_builder_fn(
      model_type="unet",
      image_shape=(None, None),
      channel_count=3,
      network_depth=5
    ):
  """
  build and return a model to the caller
  """
  from training.models import get_model_memory_usage, ConvNet, UNet

  logger.info("building model with: '%s', '%s', %i, %i" % (
      model_type, str(image_shape), channel_count, network_depth))

  model_type_fns = {
    "convnet": ConvNet,
    "unet": UNet
  }

  # FIXME: filter_sizes is also a valid param
  try:
    model = model_type_fns[model_type](
                image_shape=image_shape,
                channel_count=channel_count,
                network_depth=network_depth
            )()
  except KeyError as e:
    logger.error("'%s' is not a supported model type" % model_type)
    raise

  print(model.summary())
  mem_req = get_model_memory_usage(1, model)
  logger.info("model_size: %imb", (mem_req/1024/1024))
  return model


def model_compiler_fn(
        model,
        optimizer_name,
        loss_fn_name,
        metric_names,
        learning_rate
    ):
  """
  compile the model with an optimizer and loss function
  """
  import keras

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


def process_image(filename):
  """
  this should be merged with the training generator
  so we replicate the image input pipeline
  """
  from skimage.transform import resize
  from skimage.io import imread

  x = imread(filename)

  if len(x.shape) > 2 and x.shape[-1] not in (1,3):
    logger.warning("'%s' has wrong channel count: %i" % (filename, x.shape[-1]))
    if x.shape[-1] < 3:
      x = x.mean(axis=-1)
    elif x.shape[-1] > 3:
      x = x[:,:,:3]
    else:
      raise FileNotFoundError(filename)

  x = resize(x, (256,256), anti_aliasing=True)
  return x

def process_label(filename):
  y = process_image(filename)
  # flatten the label
  y[y > 0] = 1

  # check if the mask is multi-label
#  labels, label_counts = np.unique(i_y, return_counts=True)
#  logger.debug("label_counts: '%s'" % str(label_counts))
#  logger.debug("labels: '%s'" % str(labels))
#  # take the largest, non-zero label (zero is always background)
#  if len(labels) > 2:
#    logger.debug("dropping labels != %i" % labels[1])
#    i_y[i_y != labels[1]] = 0
#    i_y[i_y != 0] = 255
  return y[..., np.newaxis]

def write_label_image(stub, id, name, image, mask):
  from skimage.io import imread, imsave

  return

  logger.info("i_x: '%s' %f -> %f" % (str(image.shape), image.min(), image.max()))
  logger.info("i_y: '%s' %f -> %f" % (str(mask.shape), mask.min(), mask.max()))

  out = np.hstack([image, np.dstack([mask]*3)])
  logger.info("out: '%s' %f -> %f" % (str(out.shape), out.min(), out.max()))
  imsave("%s/%04i-%s" % (stub, id, name), (out*255.0).astype(np.uint8))

def image_from_filenames_generator(Xs, ys, shuffle_data=False):
  def gen():
    from skimage.transform import resize
    from skimage.io import imread, imsave
    from sklearn.utils import shuffle

    epoch_idx = 0

    while True:
      if shuffle_data:
        Xs_, ys_ = shuffle(Xs, ys)
      else:
        Xs_, ys_ = Xs, ys

      epoch_idx = epoch_idx + 1
      for X, y in zip(Xs_, ys_):
        i_x = process_image(X).astype(np.float)
        i_y = process_label(y).astype(np.float)
        write_label_image("/out/augs/val", epoch_idx, basename(X), i_x, i_y)

        yield i_x[np.newaxis, ...], i_y[np.newaxis, ...]
  return gen


def image_from_filenames_generator_with_augmentation(Xs, ys, shuffle_data=False, num_augs=5):
  """
  read images from disk with masks
  augment the images as we read them with random augmentation pipelines
  """
  import imgaug as ia
  import imgaug.augmenters as iaa

  # Define our augmentation pipeline.
  sometimes = lambda aug: iaa.Sometimes(0.5, aug)

  seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=["constant"],
            pad_cval=0
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.7, 1.3), "y": (0.7, 1.3)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.20, 0.20), "y": (-0.20, 0.20)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=0, # if mode is constant, use a cval between 0 and 255
            mode=["constant"] # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
#                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
#                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
#                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
#                iaa.SimplexNoiseAlpha(iaa.OneOf([
#                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
#                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
#                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images

# don't do dropouts
#                iaa.OneOf([
#                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
#                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
#                ]),
#                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
#                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
#                iaa.OneOf([
#                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
#                    iaa.FrequencyNoiseAlpha(
#                        exponent=(-4, 0),
#                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
#                        second=iaa.ContrastNormalization((0.5, 2.0))
#                    )
#                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
#                iaa.Grayscale(alpha=(0.0, 1.0)),
#                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
#                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
#                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
  )

#  seq = iaa.Sequential([
#      iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
#      iaa.Sharpen((0.0, 1.0)),       # sharpen the image
#      iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
#      iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
#  ], random_order=True)

  def gen():
#    from skimage.transform import resize
#    from skimage.io import imread, imsave
    from sklearn.utils import shuffle

    epoch_idx = 0

    while True:
      if shuffle_data:
        Xs_, ys_ = shuffle(Xs, ys)
      else:
        Xs_, ys_ = Xs, ys

      epoch_idx = epoch_idx + 1

      for X, y in zip(Xs_, ys_):
        i_x = process_image(X)
        i_y = process_label(y)
        write_label_image("/out/augs/trn", epoch_idx, basename(X), i_x, i_y)

        yield i_x[np.newaxis, ...], i_y[np.newaxis, ...]

        for aug_id, _ in enumerate(range(num_augs)):
          det = seq.to_deterministic()
          image_aug_i = det.augment_image(i_x)

          # make the mask rgb, then convert back and binarize
          i_y_aug = np.stack((i_y[:,:,0],) * 3, axis=-1)
          mask_aug_i = det.augment_image(i_y_aug)
          mask_aug_i = mask_aug_i.mean(axis=-1)[..., np.newaxis]
          mask_aug_i = (mask_aug_i > mask_aug_i.mean()).astype(np.uint8) * 255
#          imsave("/out/augs/trn/%s_%05i_%03i_i.png" % (basename(X), idx, aug_id), image_aug_i)
#          imsave("/out/augs/trn/%s_%05i_%03i_m.png" % (basename(y), idx, aug_id), mask_aug_i)
          yield (image_aug_i[np.newaxis, ...].astype(np.float) / 255.0,
                 mask_aug_i[np.newaxis, ...].astype(np.float) / 255.0)

  return gen


def train_model(
    modelname,
    model_builder_fn,
    model_compiler_fn,
    images,
    masks,
    batch_size,
    num_augs,
    num_epochs,
    learning_rate,
    shuffle_data,
    output_dir="models"):
  """train a simple model which just generates a mask
  """
  from keras.callbacks import (TensorBoard as TensorBoardCallback,
                               ModelCheckpoint as ModelCheckpointCallback,
                               ReduceLROnPlateau as ReduceLROnPlateauCallback,
                               EarlyStopping as EarlyStoppingCallback,
                               LearningRateScheduler)

  # OUTPUT
  try:
    makedirs(output_dir)
  except OSError:
    pass

  model_filename = join(output_dir, modelname + ".hdf5")

  # MODEL
  model = model_builder_fn()
  model_compiler_fn(model)

  # DATA
  from sklearn.model_selection import train_test_split
  train_x, test_x, train_y, test_y = train_test_split(images, masks, test_size=0.3)

  # TRAINING STEPS
  training_steps = len(train_x) // batch_size
  validation_steps = len(test_x) // batch_size

  logger.info("%i training steps, %i validation steps", training_steps, validation_steps)

  # CALLBACKS
  callbacks = [
    ModelCheckpointCallback(
        filepath=model_filename,
        verbose=0,
        save_best_only=True
    ),
#    EarlyStoppingCallback(
#        monitor="val_loss",
#        min_delta=0.00000001,
#        patience=20,
#        mode="min"
#    ),
    LearningRateScheduler(step_decay(20, 0.5))
  ]

  # TRAIN
  model.fit_generator(image_from_filenames_generator_with_augmentation(train_x, train_y, shuffle_data=shuffle_data, num_augs=num_augs)(),
                      steps_per_epoch=training_steps*(1+num_augs),
                      epochs=num_epochs,
                      validation_data = image_from_filenames_generator(test_x, test_y, shuffle_data=shuffle_data)(),
                      validation_steps = validation_steps,
                      #validation_data=validation_data,
                      callbacks=callbacks,
                      #verbose=verbosity,
                      max_queue_size=1,
                      workers=1,
                      use_multiprocessing=False)

  # FIXME: can we ditch this? the checkpoint should handle all this for us
  logger.info("saving model to '%s'" % model_filename)
  model.save(model_filename)


def get_argument_parser():
  """
  set up the argparse parser
  """
  import argparse
  import json

  parser = argparse.ArgumentParser(
      description="train network on image/mask pairs"
  )

  if "ARGS_SPEC" not in os.environ:
    logger.info("ARGS_SPEC environment variable not set, using default value")
    os.environ["ARGS_SPEC"] = "/src/train.args.spec.json"

  logger.info("parsing args.spec from '%s'" % os.environ["ARGS_SPEC"])

  # NOTE: FileNotFoundError intentionally causes termination
  with open(os.environ["ARGS_SPEC"], "r") as f:
    import builtins
    args_spec = json.load(f)
    for name, arg_spec in args_spec.items():
      logger.debug("argument: ['%s'] '%s" % (name, str(arg_spec)))
      # if the type is given, convert it to the callable function
      # from builtins (FIXME: what if it is a custom type?)
      if "type" in arg_spec:
        arg_spec["type"] = getattr(builtins, arg_spec["type"])

      parser.add_argument(name, **arg_spec)

  return parser


def parse_arguments(parser):
  """
  parse the process arguments from commandline or file
  """
  import json

  if "ARGS_FILE" in os.environ:
    # read the args from a json file
    logger.info("parsing '%s'" % os.environ["ARGS_FILE"])

    # NOTE: FileNotFoundError intentionally causes termination
    with open(os.environ["ARGS_FILE"], "r") as f:
      file_args = json.load(f)

    logger.info("args: '%s'" % json.dumps(file_args))
    args = parser.parse_args(file_args)
  else:
    # parse the args from the command line
    logger.info("parsing commandline")

    # FIXME: can't we capture and send the commandline as a param?
    #        therefore avoid the branched call here
    logger.info("args: %s", str(sys.argv))
    args = parser.parse_args()

  return args


if __name__ == "__main__":
  import sys
  import os
  from os.path import exists
  from glob import glob

  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
  ch.setFormatter(formatter)
  root.addHandler(ch)

  parser = get_argument_parser()
  args = parse_arguments(parser)

  images = sorted(glob(args.images))
  masks = sorted(glob(args.masks))

  logging.info("found %i/%i images/masks" % (len(images), len(masks)))
  assert len(images) > 0

  # get the intersection of the images and masks so we only get images with masks and vv
  images = [f for f in images if basename(f) in [basename(m) for m in masks]]
  masks  = [m for m in masks  if basename(m) in [basename(f) for f in images]]
  logging.info("found %i/%i overlapping images/masks" % (len(images), len(masks)))
  assert len(images) > 0
  assert len(images) == len(masks)


  # control memory allocation by tf
#  K.set_image_dim_ordering("tf")
  #config = tf.ConfigProto()
#  config.gpu_options.per_process_gpu_memory_fraction = 0.75
  #config.gpu_options.allow_growth = True
  #K.set_session(tf.Session(config=config))

  train_model(
      modelname=args.modelname,
      model_builder_fn=lambda: model_builder_fn(image_shape=tuple(args.image_shape), channel_count=3, network_depth=args.network_depth),
      model_compiler_fn=lambda x: model_compiler_fn(x, optimizer_name=args.optimizer, loss_fn_name=args.loss_function, metric_names=args.training_metrics, learning_rate=args.learning_rate),
      images=images,
      masks=masks,
      batch_size=args.batch_size,
      num_augs=args.num_augs,
      num_epochs=args.num_epochs,
      shuffle_data=args.shuffle_data,
      learning_rate=args.learning_rate,
      output_dir=args.output_path
  )
