from keras import layers
from keras.layers import Input, Activation
from keras.layers import Convolution3D
from keras.layers import MaxPooling3D
from keras.layers import UpSampling3D
from keras.layers import Conv3DTranspose

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D

from keras.models import Model

import tensorflow as tf

def bounce_index(start, end):
  """recursive generator to create a set of up and down
  indices from a given range,
  not including the end index
  i.e, start=0, end=5 generates [0,1,2,3,4,3,2,1,0],
  start=4, end=10, generates [4,5,6,7,8,9,8,7,6,5,4]"""
  yield start
  if start < end-1:
    for i in bounce_index(start+1, end):
      yield i
  if start < end-1:   # prevents the last element appearing twice
    yield start

class UNet():
  """class to construct a Unet given an input shape and depth"""
  def __init__(
      self,
      image_shape,
      channel_count=1,
      network_depth=6,
      filter_sizes=None,
      output_names=["output_segmentation"]):
    self.image_shape = image_shape
    self.channel_count = channel_count
    self.depth = network_depth
    self.layer_builder = UNet.build_pooled_layer
    #self.layer_builder = UNet.build_inception_layer
    self.encoder_output_name = None  # output at the bottom of the unet
    self.output_names = output_names

    if filter_sizes is None or len(filter_sizes) == 0:
      # generate a set of filter sizes
      self.filter_sizes = [2**(x+4) for x in range(0, self.depth)]
    elif len(filter_sizes) != self.depth:
      raise "filter_sizes length must match network depth"
    else:
      self.filter_sizes = filter_sizes

  def get_function_context(self):
    if len(self.image_shape) == 2:
      return {
          "conv_fn" : Convolution2D,
          "pool_fn" : MaxPooling2D,
          "trns_fn" : UpSampling2D, #Conv2DTranspose
          "link_fn" : layers.concatenate}
    elif len(self.image_shape) == 3:
      return {
          "conv_fn" : Convolution3D,
          "pool_fn" : MaxPooling3D,
          "trns_fn" : UpSampling3D, #Conv3DTranspose,
          "link_fn" : layers.concatenate}
    return {}

  @staticmethod
  def build_pooled_layer(
      layer_name,
      image_shape,
      filter_size,
      layer_stack,
      depth,
      position,
      function_context):
    """build blocks of convolution -> convolution -> pooling layers
    push and pop layers to the stack so we maintain depth linkages.
    layer_name: tf name scope to apply to the block
    image_shape: shape of the input image;
                 NB: only the length is used to determine the kernel sizes
    filter_size: number of filters to use in the convolution layers
    layer_stack: list representing a stack of layers.
                 Most recently added is at the end.
    depth:  depth of this block in the unet;
            controls whether a transpose/upsampling operation is performed
    position: index of this block in total list of blocks,
              usually total length is(2*depth)-1
    function_context: dictionary of conv, pool, trns, etc. functions;
                      so we can swap out 2d and 3d methods.
    """
    conv_size = (3, ) * len(image_shape)
    pool_size = (2, ) * len(image_shape)
    cat_axis = len(image_shape) + 1
    sigmoid_conv_size = (1, ) * len(image_shape)

    print("building '%s': %i, %i" % (layer_name, position, depth))

    if position < depth: # down
      print("  down")
      with tf.name_scope(layer_name):
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(layer_stack.pop())
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(A)
        B = function_context["pool_fn"](pool_size=pool_size, padding="same")(A)
        layer_stack.append(A)
        layer_stack.append(B)
    if position == depth: # apex
      print("  nadir")
      with tf.name_scope(layer_name):
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(layer_stack.pop())
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same", name="nadir")(A)
        layer_stack.append(A)
    if position > depth: # back up
      print("  up")
      if function_context["trns_fn"] is UpSampling2D or function_context["trns_fn"] is UpSampling3D:
        deconv = function_context["trns_fn"](pool_size)(layer_stack.pop())
      elif function_context["trns_fn"] is Conv2DTranspose or function_context["trns_fn"] is Conv3DTranspose:
        deconv = function_context["trns_fn"](filter_size, stride_size=pool_size)(layer_stack.pop())
      with tf.name_scope(layer_name):
#        A = layers.concatenate([deconv, layer_stack.pop()], axis=cat_axis)
        A = function_context["link_fn"]([deconv, layer_stack.pop()])
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(A)
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(A)
        layer_stack.append(A)

  @staticmethod
  def build_inception_layer(
      layer_name,
      image_shape,
      filter_size,
      layer_stack,
      depth,
      position,
      function_context):
    """same as pooled layer but inception layer type has a 1x1
    convolution at the begining of each block to reduce the number
    of filters from the previous layer down to 32.
    i.e., each layer has an input of 32 filters.
    """
    conv_size = (3, ) * len(image_shape)
    pool_size = (2, ) * len(image_shape)
    cat_axis = len(image_shape) + 1
    sigmoid_conv_size = (1, ) * len(image_shape)

    print("building '%s': %i, %i" % (layer_name, position, depth))

    if position < depth: # down
      print("  down")
      with tf.name_scope(layer_name):
        A = function_context["conv_fn"](32, (1, 1), activation="relu", padding="same")(layer_stack.pop())
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(A)
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(A)
        B = function_context["pool_fn"](pool_size=pool_size, padding="same")(A)
        layer_stack.append(A)
        layer_stack.append(B)
    if position == depth: # apex
      print("  nadir")
      with tf.name_scope(layer_name):
        A = function_context["conv_fn"](32, (1, 1), activation="relu", padding="same")(layer_stack.pop())
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(A)
        # FIXME: this name "nadir" is special, it is the bottom of the unet,
        # and the end of the decoder architecture,
        # we want this to be self.encoder_output_name in the unet object,
        # but because the block builder functions are static, we have no
        # reference to that object.
        # I would prefer not to expand the state of the unet object over
        # these block builder functions;
        # Maybe use internal functions with a closure around the name?
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same", name="nadir")(A)
        layer_stack.append(A)
    if position > depth: # back up
      print("  up")
      A = function_context["conv_fn"](32, (1, 1), activation="relu", padding="same")(layer_stack.pop())
      if function_context["trns_fn"] is UpSampling2D or function_context["trns_fn"] is UpSampling3D:
        deconv = function_context["trns_fn"](pool_size)(A)
      elif function_context["trns_fn"] is Conv2DTranspose or function_context["trns_fn"] is Conv3DTranspose:
        deconv = function_context["trns_fn"](filter_size, stride_size=pool_size)(A)
      with tf.name_scope(layer_name):
        A = function_context["link_fn"]([deconv, layer_stack.pop()])
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(A)
        A = function_context["conv_fn"](filter_size, conv_size, activation="relu", padding="same")(A)
        layer_stack.append(A)

  @staticmethod
  def build_residual_layer(
      layer_name,
      image_shape,
      filter_size,
      layer_stack,
      depth,
      position,
      function_context):
    """same as pooled layer but adds block outputs instead of concatenating.
    no dimension rediction is performed.
    NB: the pooled layer function can do this by changing the
    function_context to use sum instead of concatenate.
    """
    input = layer_stack.pop()
    conv_size = (3, ) * len(image_shape)
    pool_size = (2 **(position), ) * len(image_shape)

    print("%s: %s" % (layer_name, str(pool_size)))

    x = function_context["pool_fn"](pool_size=pool_size)(input)
    x = function_context["conv_fn"](32, conv_size, activation="relu", padding="same")(x)
    x = function_context["conv_fn"](32, conv_size, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(size=pool_size)(x)

  #  x = layers.add([input, x])
    x = function_context["link_fn"]([input, x])
    x = Activation("relu")(x)

    layer_stack.append(x)

  def __call__(self):
    """constructs a keras model representing this unet"""
    sigmoid_conv_size = (1, ) * len(self.image_shape)

    function_context = self.get_function_context()

    layer_stack = []

    print("UNET [%i] : %s" %(len(self.filter_sizes), str(self.filter_sizes)))

    with tf.name_scope("unet"):
      with tf.name_scope("inputs"):
        inputs = Input(self.image_shape + (self.channel_count, ))
        layer_stack.append(inputs)

  #    for i, idx in enumerate(bounce_index(0, len(self.filter_sizes))):
  #  #    if i<network_u_depth:
  #  #      dev = tf.device("/gpu:0")
  #  #    else:
  #  #      dev = tf.device("/gpu:1")
  #      dev = tf.device("/gpu:0")
  #
  #      with dev:
  #        name = "auto_layer_%i_%d" %(i, self.filter_sizes[idx])
  #        self.layer_builder(
  #          name,
  #          self.image_shape,
  #          self.filter_sizes[idx],
  #          layer_stack, self.depth-1, i, function_context)
  #
  #      # record the name of the layer at the bottom of the unet,
  #      # which corresponds to the encoder element
  #      if i == self.depth-1:
  #        self.encoder_output_name = "nadir" #layer_stack[-1].name

      # encoder
      with tf.name_scope("encoder"):
        for i, filter_size in enumerate(self.filter_sizes):
          name = "encoder_%i_%d" %(i, filter_size)
          self.layer_builder(
              name,
              self.image_shape,
              filter_size,
              layer_stack,
              self.depth-1,
              i,
              function_context)

          if i == self.depth-1:
            self.encoder_output_name = "nadir" #layer_stack[-1].name

      outputs = []

      for output_name in self.output_names:
        local_layer_stack = list(layer_stack)

        with tf.name_scope("decoder_%s" % output_name):
          for i, filter_size in enumerate(self.filter_sizes[:-1][::-1]):
            idx = len(self.filter_sizes)+i
            name = "%s_%i_%d" % (output_name, idx, filter_size)
            self.layer_builder(
                name,
                self.image_shape,
                filter_size,
                local_layer_stack,
                self.depth-1,
                idx,
                function_context)

          with tf.name_scope("output_%s" % output_name):
            # FIXME: define this as a decoder output the same way we do labels?
            outputs += [function_context["conv_fn"](1, sigmoid_conv_size, activation="sigmoid", name=output_name)(local_layer_stack.pop())]

      print("layer_stack_length : %i" % len(local_layer_stack))
      assert(len(local_layer_stack) == 0) # ensure stack is exhausted

    print("num outputs : %i" % len(outputs))

    with tf.name_scope("training_process"):
      model = Model(inputs=[inputs], outputs=outputs)

    return model
