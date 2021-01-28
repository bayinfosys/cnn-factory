"""
functions for generating crops from images

FIXME: these functions are written for 3D images, but the functions
       should generalise to 2D/nD images. Add tests and complete this
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_crop_coords(image_shape, crop_size, crop_stride=None):
  """compute the coordinates of crops for a given image and crop size
     crops are equal sized sub-regions of an image gathered at crop_stride
     spacings moving over the image.
     if the remainder is less than crop_size, the crop is taken
     from the border to guarantee equal sized crops.
     i.e., there will be an overlap in the crop sub-images.
     FIXME: this will create multiple samples where stride!=size
  """
  import itertools

  logger.debug("get_crop_coords('%s', '%s', '%s')" % (str(image_shape),
                                                      str(crop_size),
                                                      str(crop_stride)))

  if crop_stride is None:
    crop_stride = crop_size

  dims = [range(0, a, b) for a, b in zip(image_shape, crop_stride)]

  for coord in itertools.product(*dims):
    cx = list(coord)

    for i in range(len(crop_size)):
      if image_shape[i] - cx[i] < crop_size[i]:
        cx[i] = image_shape[i] - crop_size[i]

    yield (cx[0], cx[0] + crop_size[0],
           cx[1], cx[1] + crop_size[1],
           cx[2], cx[2] + crop_size[2])


def pad_image(image, requested_shape):
  """padd a crop to a requested size"""
  def get_pad_amount(smaller_region_size, larger_region_size):
    return tuple([(x-y) for x, y in zip(larger_region_size,
                                        smaller_region_size)])

  if isinstance(image, (list, tuple)):
    assert(all([len(x.shape) == len(requested_shape) for x in image]))
    assert(all([x.shape == image[0].shape for x in image]))
    pad_amount = get_pad_amount(image[0].shape, requested_shape)
  else:
    assert(len(image.shape) == len(requested_shape))
    pad_amount = get_pad_amount(image.shape, requested_shape)

  # FIXME: padd the crop if it is smaller than the crop size
  if any([p_ > 0 for p_ in pad_amount]) is True:
    right_pad = [(0, p_) for p_ in pad_amount]

    if (isinstance(image, (list, tuple))):
      return [np.pad(x, right_pad, mode="edge") for x in image]

    return np.pad(image, right_pad, mode="edge")

  # return the unpadded image
  return image


def get_crops(image, crop_coords):
  """extract sub-regions from an image
     using the given coords
  """
  assert len(image.shape) == 3

  for cc in crop_coords:
    assert len(cc) == 6
    yield image[cc[0]:cc[1], cc[2]:cc[3], cc[4]:cc[5]]


def crops_to_image(image_shape, crops, crop_size, crop_stride=None, merge_fn=None):
  """given a set of crops, repack them into the original image shape
     overlaps strategy is using the merge_fn param {None, "avg", "max", "min"}
  """
  logger.info("reconstructing input from %i crops" % len(crops))
  #crop_coords = list(self.get_crop_coords(self.template_image_shape, crop_size))
  #logger.info("got %i crop coords" % len(crop_coords))

  img = np.zeros(image_shape, dtype=np.float32)
  cnt = np.zeros(image_shape, dtype=np.int16)
  logger.info("filling image.shape: '%s'" % str(img.shape))

  for crop, crop_coord in \
        zip(crops, get_crop_coords(image_shape, crop_size, crop_stride)):
    try:
      assert len(crop.shape) == 3
      assert len(crop_coord) == 6
      assert len(crop_coord) == 2*len(crop.shape)

      src = img[crop_coord[0]:crop_coord[1],
                crop_coord[2]:crop_coord[3],
                crop_coord[4]:crop_coord[5]]
      dst = crop

      if merge_fn == "avg" or merge_fn == "sum":
        dst = src + dst
      elif merge_fn == "min":
        dst = np.minimum(src, dst)
      else: #elif merge_fn == "max":
        dst = np.maximum(src, dst)

      # save the result
      img[crop_coord[0]:crop_coord[1],
          crop_coord[2]:crop_coord[3],
          crop_coord[4]:crop_coord[5]] = dst
      # accumulate the overlaps
      cnt[crop_coord[0]:crop_coord[1],
          crop_coord[2]:crop_coord[3],
          crop_coord[4]:crop_coord[5]] += 1

    except Exception as e:
      print(e)
      logger.exception(e)
      continue

  # apply totalling
  if merge_fn == "avg":
    img = img/cnt

  return img
