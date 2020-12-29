"""
custom loss functions

see:
https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou/276144#276144
"""

from keras import backend as K
from keras.losses import binary_crossentropy

import tensorflow as tf

def dice_coef(y_true, y_pred):
  """compute the dice coefficient of two classifications
  https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

  NB:
  However, having a larger smooth value (also known as Laplace smooth,
  or Additive smooth) can be used to avoid overfitting. The larger the smooth
  value the closer the following term is to 1 (if everything else is fixed),
  https://github.com/pytorch/pytorch/issues/1249

  NB: we take the sum of squares in the union to be more sensitive to outliers
  """
  with tf.name_scope("dice_coef"):
    #smooth = 1.0
    smooth = 1e-07
    thresh = 0.5

#    y_pred = K.maximum(y_pred, thresh)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)


def dice_coef_batch(y_true, y_pred):
  """compute the dice coefficient over batches
  """
  with tf.name_scope("dice_coef"):
#    smooth = 1.0
    smooth = 1e-07
    thresh = 0.5
    #non_batch_axes = K.slice(K.shape(y_true), 1, -1)
#    non_batch_axes = (1,2,3,4)
    non_batch_axes = (1,2,3)

#    y_pred = K.cast (y_pred > thresh, dtype = tf.float32)

    intersection = K.sum(y_true * y_pred, axis=non_batch_axes)
    union = K.sum(y_true, axis=non_batch_axes) + K.sum(y_pred, axis=non_batch_axes)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
  """dice loss
  sometimes called smooth dice
  1.0 - dice
  We don't take -dice incase the values sum to greater than one,
  in that case the error will be visible in the training process.

  the original ultrasound repo returned -dice
  https://github.com/jocicmarko/ultrasound-nerve-segmentation

  for multi-label images:
  return K.mean (1.0-dice_coef (y_true, y_pred), axis=-1) # average across batch

  we may also try the reciprocal
  https://github.com/DeepCognition/ultrasound-nerve-segmentation/blob/master/train.py
  smooth = 1.0
  return 1.0 / (dice_coef (y_true, y_pred) + smooth)
  (no benefit)

  or log dice:
  -K.log (dice_coef (y_true, y_pred))
  (no benefit)
  """
  with tf.name_scope("dice_coef_loss"):
#    return 1.0-dice_coef(y_true, y_pred)
    return 1.0-dice_coef_batch(y_true, y_pred)

def binary_cross_entropy_plus_dice_loss(y_true, y_pred):
  """add bce and dice to get some more error
  """
  with tf.name_scope("bce_dice_loss"):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

#def focal_loss(y_true, y_pred):
#  """
#  Focal Loss for Dense Object Detection
#  Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
#  https://arxiv.org/abs/1708.02002
#  https://github.com/mkocabas/focal-loss-keras
#  """
#  alpha = 0.25
#  gamma = 2.0
#
#  with tf.name_scope("focal_loss"):
#    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#
#    A = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
#    B = -K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#    return A+B

def jaccard_index(y_true, y_pred, smooth=100.):
  """jaccard index as an alternative to dice.
  https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
  because don't have that version of keras yet
  """
  non_batch_axes = tuple(K.shape(y_true)[1:])
  intersection = K.sum(K.abs(y_true * y_pred), axis=non_batch_axes)
  sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=non_batch_axes)
  return K.mean((intersection + smooth) / (sum_ - intersection + smooth), axis=0)

def jaccard_index_loss(y_true, y_pred):
  """jaccard loss as 1.0 - jaccard distance
  """
  return 1.0 - jaccard_index(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth=1):
  """intersection-over-union
  """
  non_batch_axes = tuple(y_true.shape[1:])
  intersection = K.sum(K.abs(y_true * y_pred), axis=non_batch_axes)
  union = K.sum(y_true, axis=non_batch_axes) + K.sum(y_pred, axis=non_batch_size)-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou
