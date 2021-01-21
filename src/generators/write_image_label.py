import os
import logging

import numpy as np


from os.path import join
from skimage.io import imsave


def write_image_label(stub, id, name, image, mask):
  """
  combine the image and label horizontally and write to disk
  """
  logging.info("i_x: '%s' %f -> %f" % (str(image.shape), image.min(), image.max()))
  logging.info("i_y: '%s' %f -> %f" % (str(mask.shape), mask.min(), mask.max()))

  out = np.hstack([image, np.dstack([mask]*3)])
  logging.info("out: '%s' %f -> %f" % (str(out.shape), out.min(), out.max()))
  imsave(join(stub, "%04i-%s" % (id, name)), (out*255.0).astype(np.uint8))
