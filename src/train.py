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


def build_model(image_shape=(None, None), num_outputs=1):
  """build and return a model to the caller
  """
  from training.models import get_model_memory_usage, ConvNet, UNet

#  model = ConvNet(
#      image_shape=image_shape,
#      channel_count=3,
#      network_depth=3,
#  )()

  model = UNet(
      image_shape=image_shape,
      channel_count=3,
      network_depth=6,
#      filter_sizes=[32]*4,
  )()

  print(model.summary())
  mem_req = get_model_memory_usage(1, model)
  logger.info("model_size: %imb", (mem_req/1024/1024))
  return model


def compile_model(model, **kwargs):
  """compile the model with an optimizer and loss function
  """
  import keras.optimizers

  optimizer = keras.optimizers.Adam(lr=kwargs.get("learning_rate", 0.0001))

  # compile the model
  model.compile(
      optimizer=optimizer,
      loss=bce_dice,
#      loss=dice_coef_loss
#      loss="binary_crossentropy",
#      loss=jaccard_index_loss,
      metrics=[dice_coef]
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
    model_name,
    model_builder,
    model_compiler,
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

  model_filename = join(output_dir, model_name + ".hdf5")

  # MODEL
  model = build_model((256, 256))
  compile_model(model, learning_rate=learning_rate)

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

  logger.info("saving model to '%s'" % model_filename)
  model.save(model_filename)


if __name__ == "__main__":
  import argparse
  import json
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

  parser = argparse.ArgumentParser(
      description="train network on image/mask pairs"
  )
  parser.add_argument("--modelname", default="unamed", help="name")
  parser.add_argument("--batch-size", "-b", default=8, type=int, help="batch size")
  parser.add_argument("--num-augs", default=0, type=int, help="number of augmentations")
  parser.add_argument("--num-epochs", "-e", default=100, type=int, help="number of epochs")
  parser.add_argument("--shuffle-data", action="store_true")
  parser.add_argument("--learning-rate", default=0.0001, type=float)
  parser.add_argument("--output-path", default="models", help="output dir")
  parser.add_argument("--images", required=True)
  parser.add_argument("--masks", required=True)
  args = parser.parse_args()

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
      model_name=args.modelname,
      model_builder=build_model,
      model_compiler=compile_model,
      images=images,
      masks=masks,
      batch_size=args.batch_size,
      num_augs=args.num_augs,
      num_epochs=args.num_epochs,
      shuffle_data=args.shuffle_data,
      learning_rate=args.learning_rate,
      output_dir=args.output_path
  )
