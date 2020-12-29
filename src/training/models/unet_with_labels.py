import tensorflow as tf
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import GlobalMaxPooling2D
from keras.models import Model

from .unet import UNet

class UNetWithLabels():
  """class to construct a Unet with labeled output
  Constructs a standard unet and appends some flatten and dense layers
  to the bottom element such that a label can be predicted from the
  internal rep.
  The label-predictors are described by the label definitions dict,
  where each entry describes an output of the model.
  {
    "output1" : {"type":"category", "size":10, "weight": 1.0}
    "output2" : {"type":"numeric", "size":1, "weight":0.5},
    "output3" : {"type":"segmentation", "size":1, "weight":0.2}
  }
  "type":
    "category" types used softmax output with size "count"
    "numeric" types use relu outputs
    "segmentation" types are image outputs
    FIXME: include warp image types, for image to image mapping?
  "size": the number of unique output values, i.e., softmax vector length
  "weight": contribution of this outputs loss to the total loss value
  """
  def __init__(
      self,
      image_shape,
      network_depth=6,
      filter_sizes=[],
      label_definitions={}):
    # get the segmentation outputs from the label defs
    seg_outputs = []
    for k in label_definitions:
      if label_definitions[k]["type"] == "segmentation":
        seg_outputs += [k]

    self.unet = UNet(
        image_shape=image_shape,
        network_depth=network_depth,
        filter_sizes=filter_sizes,
        output_names=seg_outputs)

    self.label_defs = label_definitions

  def __call__(self):
    """constructs a keras model representing this label unet"""
    with tf.name_scope("unet_with_labels"):
      model = self.unet()

      names = [l for l in self.label_defs.keys()
               if self.label_defs[l]["type"] != "segmentation"]


      # create a bunch of classification blocks for labels
      for name in names:
        label_def = self.label_defs[name]
        encoder = model.get_layer(self.unet.encoder_output_name).output

        with tf.name_scope("decoder_%s" % name):
          M = Convolution2D(64, kernel_size=(1, 1), padding="same", activation="relu")(encoder)

          # set the output depending on the label type
          if label_def["type"] == "category":
            M = Convolution2D(label_def["size"], kernel_size=(1, 1), padding="same", activation="relu")(M)
            M = GlobalMaxPooling2D()(M)
            output = Dense(label_def["size"], activation="softmax", name=name)(M)
          elif label_def["type"] == "numeric":
            M = Convolution2D(32, kernel_size=(1, 1), padding="same", activation="relu")(M)
            M = GlobalMaxPooling2D()(M)
            output = Dense(1, activation="relu", name=name)(M)
          #elif label_def["type"] == "segmentation":
          #  # we currently build the decoder in the unet class...
          #  # don't build a new output, keras will baulk
          #  continue
          else:
            assert (False)

          #if label_def["type"] == "category":
          #  M = Flatten()(M)
          #  M = Dense(1024, activation="relu")(M)
          #  O = Dense(L["size"], activation="softmax", name=name)(M)
          #elif label_def["type"] == "numeric":
          #  M = Flatten()(M)
          #  M = Dense(1024, activation="relu")(M)
          #  M = Dense(50, activation="relu")(M)
          #  O = Dense(1, activation="relu", name=name)(M)
          ##elif label_def["type"] == "segmentation":
          ##  continue
          #else:
          #  raise

        model.outputs.append(output)

    print("LUnet with %i outputs" % (len(model.outputs)))

    # construct a new model otherwise the output names are incorrect
    return Model(inputs=model.inputs, outputs=model.outputs)
