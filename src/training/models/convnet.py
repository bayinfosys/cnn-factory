import numpy as np
from keras import layers
from keras.layers import Input, Activation
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import concatenate
#from keras.layers import Dropout
from keras.models import Model

import tensorflow as tf

class ConvNet ():
  """constructor class for a convolutional neural network
  """
  def __init__ (self, image_shape, channel_count = 1, network_depth = 3):
    self.image_shape = image_shape
    self.image_channel_count = channel_count
    self.depth = network_depth

  def __call__ (self):
    """ basic U-net style implementation of a convnet
        for semantic-segmentation
    """
    if self.depth == 3:
      return self.__3_down()
    elif self.depth == 2:
      return self.__2_down()
    else:
      raise NotImplementedError("depth of %i not implemented" % self.depth)

  def __3_down(self):
    f_sz = [16, 32, 64]
    k_sz = (3,) * len (self.image_shape) # kernel size
    p_sz = (2,) * len (self.image_shape) # pool size

    conv_p = {"activation": "relu", "padding": "same", "kernel_size": k_sz}
    trns_p = {"activation": "relu", "padding": "same", "kernel_size": p_sz,"strides": p_sz}
    pool_p = {"pool_size": p_sz, "padding": "valid"}
    upsm_p = {"size": (2,) * len(self.image_shape)}

    # input block
    filter_size=f_sz[0]
    inputs = Input ((*self.image_shape, self.image_channel_count))
    x = Convolution2D (filter_size, **conv_p) (inputs)
    x = Convolution2D (filter_size, **conv_p) (x)

    # down-scale
    for filter_size in f_sz:
      x = MaxPooling2D (**pool_p) (x)
      x = Convolution2D (filter_size, **conv_p) (x)
      x = Convolution2D (filter_size, **conv_p) (x)

    # up-scale
    for filter_size in f_sz[::-1]:
#     x = Conv2DTranspose (filter_size, **trns_p) (x)
      x = UpSampling2D (**upsm_p) (x)
      x = Convolution2D (filter_size, **conv_p) (x)
      x = Convolution2D (filter_size, **conv_p) (x)

    # output
    output = Convolution2D (1, (1,1), activation="sigmoid") (x)

    return Model (inputs=[inputs], outputs=[output])


#    inputs = Input ((*self.image_shape, self.image_channel_count))
#
#    A = Convolution2D (f_sz[0], **conv_p) (x)
#    A = Convolution2D (f_sz[0], **conv_p) (A)
#    A = MaxPooling2D (**pool_p) (A)
#
#    B = Convolution2D (f_sz[1], **conv_p) (A)
#    B = Convolution2D (f_sz[1], **conv_p) (B)
#    B = MaxPooling2D (**pool_p) (B)
#
#    C = Convolution2D (f_sz[2], **conv_p) (B)
#    C = Convolution2D (f_sz[2], **conv_p) (C)
#    C = MaxPooling2D (**pool_p) (C)
#
###    E = Conv2DTranspose (f_sz[-2], **trns_p) (C)
#    E = UpSampling2D (**upsm_p) (C)
#    E = Convolution2D (f_sz[-1], **conv_p) (E)
#    E = Convolution2D (f_sz[-1], **conv_p) (E)
#
###    F = Conv2DTranspose (f_sz[-3], **trns_p) (E)
#    F = UpSampling2D (**upsm_p) (E)
#    F = Convolution2D (f_sz[-2], **conv_p) (F)
#    F = Convolution2D (f_sz[-2], **conv_p) (F)
#
###    G = Conv2DTranspose (f_sz[-4], **trns_p) (F)
#    G = UpSampling2D (**upsm_p) (F)
#    G = Convolution2D (f_sz[-3], **conv_p) (G)
#    G = Convolution2D (f_sz[-3], **conv_p) (G)
#
#    output = Convolution2D (1, (1,1), activation="sigmoid") (G)
#
#    return Model (inputs=[inputs], outputs=[output])


  def __2_down(self):
    """ basic U-net style implementation of a convnet
        for semantic-segmentation
    """
    k_sz = (2,) * len (self.image_shape) # kernel size
    p_sz = (2,) * len (self.image_shape) # pool size

    conv_p = {"activation": "relu", "padding": "same", "kernel_size": k_sz}
    trns_p = {"activation": "relu", "padding": "same", "kernel_size": p_sz,"strides": p_sz}
    pool_p = {"pool_size": p_sz, "padding": "valid"}

    inputs = Input ((*self.image_shape, self.image_channel_count))

    A = Convolution2D (32, **conv_p) (inputs)
    A = Convolution2D (32, **conv_p) (A)
    A = MaxPooling2D (**pool_p) (A)

    B = Convolution2D (64, **conv_p) (A)
    B = Convolution2D (64, **conv_p) (B)
    B = MaxPooling2D (**pool_p) (B)

    E = Conv2DTranspose (64, **trns_p) (B)
    E = Convolution2D (64, **conv_p) (E)
    E = Convolution2D (64, **conv_p) (E)

    F = Conv2DTranspose (32, **trns_p) (E)
    F = Convolution2D (32, **conv_p) (F)
    F = Convolution2D (32, **conv_p) (F)

    output = Convolution2D (1, (1,1), activation="sigmoid") (F)

    return Model (inputs=[inputs], outputs=[output])

