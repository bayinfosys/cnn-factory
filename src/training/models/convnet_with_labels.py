import numpy as np

from tensorflow.keras.layers import (Input, Activation,
                                     Convolution2D,
                                     AveragePooling2D, MaxPooling2D,
                                     Conv2DTranspose, UpSampling2D,
                                     GlobalAveragePooling2D,
                                     Dropout)

from tensorflow.keras.models import Model

from .convnet import ConvNet


class ConvNetWithLabels ():
  def __init__ (self, image_shape, network_depth = 3, label_definitions = []):
    self.convnet = ConvNet (image_shape=image_shape, network_depth=network_depth)
    #self.image_channel_count = 3
    self.label_defs = label_definitions

  def __call__ (self):
    model = self.convnet ()

    # create a bunch of classification blocks for labels
    for L in self.label_defs:
      C = model.get_layer ("encoder_output").output #self.unet.encoder_output)
      M = Convolution2D (32, kernel_size=(1,1), padding="same", activation="relu") (C)

      # set the output depending on the label type
      if L["type"] == "category":
        M = Convolution2D (L["size"], kernel_size=(1,1), padding="same", activation="softmax") (C)
      elif L["type"] == "numeric":
        M = Convolution2D (32, kernel_size=(1,1), padding="same", activation="relu") (C)
        M = Convolution2D (1, kernel_size=(1,1), padding="same", activation="relu") (C)
      else:
        raise

      O = GlobalAveragePooling2D (name=L["name"]) (M)

      model.outputs.append (O)

    # construct a new model otherwise the output names are incorrect
    return Model (inputs = model.inputs, outputs = model.outputs)
