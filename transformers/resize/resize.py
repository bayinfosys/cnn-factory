import os
import sys
import logging

import numpy as np

from os.path import join, basename

from glob import glob

logger = logging.getLogger(__name__)

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def resize(input_file, output_file, new_size):
  """
  transform the file by resizing
  """
  from skimage.io import imread, imsave
  from skimage.transform import resize

  A = imread(input_file).astype(np.float)/255.0
  x = resize(A, new_size, anti_aliasing=True)

  try:
    imsave(output_file, (x*255.0).astype(np.uint8), check_contrast=False)
  except IOError as e:
    logger.error("%s: %s" % (input_file, str(e)))


if __name__ == "__main__":
  import argparse
  from time import perf_counter as clock

  log_format = "[%(asctime)s] - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

  root = logging.getLogger()
  root.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

  ch = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter(log_format)
  ch.setFormatter(formatter)
  root.addHandler(ch)

  parser = argparse.ArgumentParser(description="resize images")
  parser.add_argument("--images", "-i", help="glob pattern of filenames for the images")
  parser.add_argument("--output", "-o", help="output directory")
  parser.add_argument("--size", "-s", type=int, nargs="+", default=(256, 256))
  parser.add_argument("--dry-run", "-n", action="store_true", default=False, help="dry run without any changes")

  args = parser.parse_args()

  T = clock()
  images = sorted(glob(args.images))
  logger.info("found %i images in %0.2fs" % (len(images), (clock()-T)))

  T = clock()
  o_images = [join(args.output, basename(x)) for x in images]
  logger.info("renamed %i images in %0.2fs" % (len(o_images), (clock()-T)))

  T = clock()
  for idx, (i, o) in enumerate(zip(images, o_images)):
    logger.debug("converting %i/%i [%s, %s]" % (idx, len(images), i, o))

    if args.dry_run is False:
      resize(i, o, new_size=args.size)

  logger.info("converted %i images in %0.2fs" % (len(images), (clock()-T)))
