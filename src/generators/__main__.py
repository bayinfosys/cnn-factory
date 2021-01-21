"""
top level function to run the generators and augmentations
outside a training process
"""
import os
import sys
import logging
import argparse

import numpy as np

from glob import glob
from time import perf_counter as clock

from skimage.io import imread
from skimage.transform import resize

from .image_from_filenames import create_image_from_filenames_generator
from .write_image_label import write_image_label


def setup_logging():
  log_format = "[%(asctime)s] - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

  logger = logging.getLogger(__name__)
  logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

  ch = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter(log_format)
  ch.setFormatter(formatter)
  logger.addHandler(ch)


def setup_argparse():
  parser = argparse.ArgumentParser(description="generate and augment images")
  parser.add_argument("--images", "-i", help="glob pattern of filenames for the images")
  parser.add_argument("--labels", "-l", help="glob pattern of filenames for the label images")
  parser.add_argument("--output", "-o", help="output directory")
  parser.add_argument("--size", "-s", type=int, nargs="+", default=(256, 256), help="image size")
  parser.add_argument("--num-augs", "-a", type=int, default=0, help="number of augmentations to produce")
  parser.add_argument("--dry-run", "-n", action="store_true", default=False, help="dry run without any output")
  return parser


if __name__ == "__main__":
  setup_logging()
  parser = setup_argparse()
  args = parser.parse_args()

  logger = logging.getLogger(__name__)

  # read the image filenames
  images = sorted(glob(args.images))
  labels = sorted(glob(args.labels))
  logger.info("found %i/%i images/labels" % (len(images), len(labels)))

  assert len(images) > 0
  assert len(labels) > 0

  def image_loader(filename):
    x = imread(filename).astype(np.float) / 255.0
    x = resize(x, tuple(args.size), anti_aliasing=True)
    return x

  def label_loader(filename):
    x = imread(filename).astype(np.uint8)
    x = (x==1).astype(np.float)
    x = resize(x, tuple(args.size), anti_aliasing=False)
    return x

  def image_validator(x):
    return ((len(x.shape) == 3) and
            (x.shape[0] == args.size[0]) and
            (x.shape[1] == args.size[1]) and
            (x.shape[2] == 3))

  def label_validator(x):
    return ((len(x.shape) == 2) and
            (x.shape[0] == args.size[0]) and
            (x.shape[1] == args.size[1]))

  # create a generator
  gen = create_image_from_filenames_generator(
      images,
      labels,
      image_preprocess_fn=image_loader,
      label_preprocess_fn=label_loader,
      image_validation_fn=image_validator,
      label_validation_fn=label_validator,
      augmentation_fn=None,
      debug_output_path=None
  )

  for index, (image, label) in enumerate(gen()):
    logger.info("image: '%s:%s' [%0.2f -> %0.2f], label: '%s:%s' [%0.2f -> %0.2f]" % (
        str(image.shape), str(image.dtype), image.min(), image.max(),
        str(label.shape), str(label.dtype), label.min(), label.max(),
    ))

    if args.dry_run is False:
      write_image_label(args.output, index, "basic.png", image[0, ...], label[0, ...])
