import logging

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


logger = logging.getLogger(__name__)


def create_image_augmentation_fn(num_augs = 4):
  """
  create an augmentation pipeline and return a function
  which applies that pipeline to a pair of images, yielding
  the augmentations

  uses imgaug to perform augmentations, which depends on opencv
  https://imgaug.readthedocs.io/

  num_augs: number of augmentations to perform in the generator
  """
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

  def augmentation(X, y):
    for aug_id, _ in enumerate(range(num_augs)):
      det = seq.to_deterministic()
      aug_x = det.augment_image(X)

      # make the mask rgb, then convert back and binarize
      _y = np.stack((y[:,:,0],) * 3, axis=-1)
      aug_y = det.augment_image(_y)
      aug_y = aug_y.mean(axis=-1)[..., np.newaxis]
      aug_y = (aug_y > aug_y.mean()).astype(np.uint8) * 255

      # logger.debug("yielding: '%s'/'%s'" % (str(i_x.shape), str(i_y.shape)))
      yield (aug_x.astype(np.float) / 255.0,
             aug_y.astype(np.float) / 255.0)

  if num_augs == 0:
    return None

  return augmentation
