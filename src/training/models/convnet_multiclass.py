import numpy as np
from keras import layers
from keras.layers import Input, Activation
from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import concatenate
#from keras.layers import Dropout
from keras.models import Model

import tensorflow as tf

class ConvNetMultiClass ():
  """constructor class for a multi-class convolutional neural network
  """
  def __init__ (self, image_shape, channel_count = 1, network_depth = 3, outputs = 1):
    self.image_shape = image_shape
    self.image_channel_count = channel_count
    self.depth = network_depth
    self.outputs = outputs

  def __call__ (self):
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

    C = Convolution2D (128, **conv_p) (B)
    C = Convolution2D (128, **conv_p) (C)
    C = MaxPooling2D (**pool_p) (C)

    outputs = []
    for output_id in range(self.outputs):
#      D = Conv2DTranspose (128, **trns_p) (C)
      D = UpSampling2D (**pool_p) (C)
      D = Convolution2D (128, **conv_p) (D)
      D = Convolution2D (128, **conv_p) (D)

#      E = Conv2DTranspose (64, **trns_p) (D)
      E = UpSampling2D (64, **pool_p) (D)
      E = Convolution2D (64, **conv_p) (E)
      E = Convolution2D (64, **conv_p) (E)

#      F = Conv2DTranspose (32, **trns_p) (E)
      F = UpSampling2D (**pool_p) (E)
      F = Convolution2D (32, **conv_p) (F)
      F = Convolution2D (32, **conv_p) (F)

      output = Convolution2D (1, (1,1), activation="sigmoid") (F)
      outputs.append(output)

    return Model (inputs=[inputs], outputs=outputs)
