"""
top level function to run the generators and augmentations
outside a training process

Simply loads the inputs and labels of a data set and prints
the shape to stdout.
If the input is a filename input_loader attempts to read the
file as an image.
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
from .csv_generator import csv_to_lists


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
  parser.add_argument("--csv", help="filename of csv containing inputs and outputs")
  parser.add_argument("--inputs", nargs="+", help="csv column names of input data")
  parser.add_argument("--outputs", nargs="+", help="csv column names of output data")
  parser.add_argument("--images", "-i", help="glob pattern of filenames for the images")
  parser.add_argument("--labels", "-l", help="glob pattern of filenames for the label images")
  parser.add_argument("--output", "-o", help="output directory")
  parser.add_argument("--size", "-s", type=int, nargs="+", default=(256, 256), help="image size")
  parser.add_argument("--num-augs", "-a", type=int, default=0, help="number of augmentations to produce")
  parser.add_argument("--dry-run", "-n", action="store_true", default=False, help="dry run without any output")
  return parser


def input_loader(xs):
  """load input data"""
  from os.path import exists

  i = []

  for x in xs:
    if isinstance(x, str) and exists(x):
      v = imread(x).astype(np.float) / 255.0
      v = resize(v, tuple(args.size), anti_aliasing=False)
    else:
      v = np.array(x)

    i.append(v)

  i = np.array(i).astype(np.float) if len(i) > 1 else i[0]

  logger.debug("input.shape: '%s:%s'" % (str(i.shape), str(i.dtype)))
  return i


def label_loader(ys):
  """load a label file"""
  from os.path import exists

  l = []

  for y in ys:
    if isinstance(y, str) and exists(y):
      v = imread(y).astype(np.uint8)
      v = (v==1).astype(np.float)
      v = resize(v, tuple(args.size), anti_aliasing=False)
    else:
      v = np.array(y)

    l.append(v)

  l = np.array(l).astype(np.float) if len(l) > 1 else l[0]

  logger.debug("label.shape: '%s:%s'" % (str(l.shape), str(l.dtype)))
  return l


def input_validator(x):
  return x is not None

def label_validator(x):
  return x is not None

if __name__ == "__main__":
  setup_logging()
  parser = setup_argparse()
  args = parser.parse_args()

  logger = logging.getLogger(__name__)

  if args.csv is not None:
    data = csv_to_lists(args.csv)
    logger.info("found %i/%i rows/columns" % (len(list(data.items())[0]), len(data)))

    input_keys = args.inputs
    output_keys = args.outputs
    assert input_keys is not None, "require input column names when using csvfile"
    assert output_keys is not None, "require output column names when using csvfile"

    # create sets of tuples for the inputs and ouputs
    inputs = list(zip(*[data[k] for k in input_keys]))
    outputs = list(zip(*[data[k] for k in output_keys]))
  elif args.images is not None and args.labels is not None:
    # read the image filenames
    inputs = [(x,) for x in sorted(glob(args.images))]
    outputs = [(x,) for x in sorted(glob(args.labels))]
  else:
    raise NotImplementedError("Cannot generate data from nothing")

  logger.info("found %i/%i inputs/outputs" % (len(inputs), len(outputs)))
  assert len(inputs) > 0
  assert len(outputs) > 0
  assert len(inputs) == len(outputs)

  # create a generator
  gen = create_image_from_filenames_generator(
      inputs,
      outputs,
      image_preprocess_fn=input_loader,
      label_preprocess_fn=label_loader,
      image_validation_fn=input_validator,
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
