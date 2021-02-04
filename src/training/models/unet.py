import json
import logging

import tensorflow as tf

from tensorflow.keras.layers import (Input, Activation,
                                     Convolution3D,
                                     MaxPooling3D, UpSampling3D, Conv3DTranspose,
                                     Convolution2D,
                                     MaxPooling2D, Conv2DTranspose, UpSampling2D,
                                     concatenate, add,
                                     Flatten, Dense, Reshape,
                                     GlobalAveragePooling2D, GlobalMaxPool2D,
                                     GlobalAveragePooling3D, GlobalMaxPool3D)

from tensorflow.keras.models import Model


logger = logging.getLogger(__name__)


class UNet():
  """
  class to construct a Unet

  takes as input:
    network_depth: number of layers to create in the network
    input_definitions: map of input names to definitions
    output_definitions: [optional] map of output names to definitions

  input_definitions is a map of input_definition:
  ```
  {
    "type": (category|numeric|segmentation|image)
    "shape": shape of the input data array; images must have channel as the last entry
  }
  ```

  output_definitions is a map of output_definition:
  ```
  {
    "type": (category|numeric|segmentation|image),
    "size": <number of outputs of this type>,
    "weight": <0..1, contribution to the loss function>
  }
  ```

  multiple outputs maybe specified, i.e.:
  ```
  {
    "class_label_output": {"type":"category", "size":10, "weight": 1.0}
    "number_value_output": {"type":"numeric", "size":1, "weight":0.5},
    "semantic_map_output" : {"type":"segmentation", "size":1, "weight":0.2}
  }
  ```

  `type`: (category|numeric|segmentation|image)
    `category`: dense layer with `size` outputs`, (number of categories in the one-hot), relu activation
    `numeric`: dense layer with `size` outputs, feeding a single output layer, relu activation
    `segmentation`: image output with single channel, sigmoid activation
    `image`: image output with network_depth channels, sigmoid activation
  """
  def __init__(
      self,
      network_depth=6,
      filter_sizes=None,
      input_definitions=None,
      output_definitions=None):
    """
    initialise the variables in the model.

    filter_sizes: number of filters at each layer in the network; the last element in this array
                  defines the size of the latent space, and the reversed array defines the decoder
                  filter progression.

    input_definitions: map of input names to input definitions,
                       if none is specified, the input is assumed to be a 128x128x3 channel image
    output_definitions: map of output names to output definitions
                        if none is specified, the default segmentation output is assumed
    """
#    self.image_shape = image_shape
#    self.channel_count = channel_count
    self.depth = network_depth
    self.decoder_type = "upsample" # (upsample|convolution)
    self.latent_space_mode = "normal" # (normal|vae)
    self.layer_link_type = "concatenate" # (none|concatenate|add)

    # FIXME: create a list of encoder_filters, latent_filters, and decoder_filters
    self.filter_sizes = [2**(x+4) for x in range(0, self.depth)] if filter_sizes is None else filter_sizes

    if len(self.filter_sizes) != self.depth:
      raise ValueError("filter_sizes length must match network depth (%i != %i)" % (len(self.filter_sizes), self.depth))

    if input_definitions is None:
      logger.warning("no input definitions; using default")
      self.encoder_definitions = {"input": {"type": "image", "shape": (256, 256, 3), "weight": 1.0}}
    else:
      logger.info("input_definitions: '%s'" % json.dumps(input_definitions))
      self.encoder_definitions = input_definitions

    if output_definitions is None:
      logger.warning("no output definitions; using default")
      self.decoder_definitions = {"output": {"type": "segmentation", "size": 1, "weight": 1.0}}
    else:
      logger.info("output_definitions: '%s'" % json.dumps(output_definitions))
      self.decoder_definitions = output_definitions

    # layer parameterisation
    # FIXME: construct these parameters (and the function_context) into
    #        a large map object which can be passed around, serialised,
    #        and mocked for unit testing the block functions.
    #self.conv_size = (3, ) * len(self.image_shape)
    #self.pool_size = (2, ) * len(self.image_shape)
    #self.sigmoid_conv_size = (1, ) * len(self.image_shape)

    self.encoder_conv_params = {
        #"kernel_size": self.conv_size,
        "activation": "relu",
        "padding": "same"
    }

    self.encoder_pool_params = {
        #"pool_size": self.pool_size,
        "padding": "same"
    }

    self.latent_params = {
        "activation": "relu"
    }

    # self.output_conv_kernel = (1, ) * len(self.image_shape)

    self.output_conv_params = {
        "filters": 32,
        # "kernel_size": self.output_conv_kernel,
        "activation": "relu",
        "padding": "same"
    }

    self.function_context = self._get_function_context()

  def _get_function_context(self):
    """
    create a map of functions to use for layer building operations.
    if we are working on 2D images, use 2D layers,
    if we are working on 3D images, use 3D layers

    decoder_type: type of layer to use in the decoder stage for restoring image size
                  `upsample` uses upsampling layers
                  `convolution` uses transposed convolutional layers
    layer_link_type: type of connection between layers at different scales:
                     `none` no connecting of encoder and decoder layers (typical CNN)
                     `concatenate` joins the two layers in the UNet style
                     `add` sums the two layers in the resnet style (this
                     requires a skip architecture which is not implemented)
    latent_space_mode: layer type to use for the latent space between the
                       encoder and decoder stages.
                       `normal` is a convolutional layer as used elsewhere,
                       `vae` is a dense layer (mlp) as used in vae architectures.
    NB: here, we can change the link_fn to "sum" to build a resnet
    """
    link_layers = {
        "none": lambda l: l[0],
        "concatenate": concatenate,
        "add": add
    }

    # FIXME: 1D convolution
    # FIXME: this needs to be adjusted for the shape of the encoder/decoder branch
    #        we are running on (maybed 2D and 3D input, 1D output, etc)
    HACK_DIMS = self.encoder_definitions["image"]["shape"][:-1]

    if len(HACK_DIMS) == 2:
      return {
          "conv": Convolution2D,
          "pool": MaxPooling2D,
          "trns": UpSampling2D if self.decoder_type == "upsample" else Conv2DTranspose,
          "link": link_layers[self.layer_link_type],
          "latent": Convolution2D if self.latent_space_mode == "normal" else Dense,
          "global": GlobalAveragePooling2D,
      }
    elif len(HACK_DIMS) == 3:
      return {
          "conv": Convolution3D,
          "pool": MaxPooling3D,
          "trns": UpSampling3D if self.decoder_type == "upsample" else Conv3DTranspose,
          "link": link_layers[self.layer_link_type],
          "latent": Convolution3D if self.latent_space_mode == "normal" else Dense,
          "global": GlobalAveragePooling3D,
      }

    else:
      raise NotImplementedError("Cannot build function context for %iD images" % len(HACK_DIMS))

  def encoder_block(self, filter_size, dims, layer_stack):
    """
    construct an encoder block
      conv
      conv
      pool
    """
    conv_params = {}
    conv_params.update(self.encoder_conv_params)
    conv_params["filters"] = filter_size
    conv_params["kernel_size"] = (3, ) * dims

    pool_params = {}
    pool_params.update(self.encoder_pool_params)
    pool_params["pool_size"] = (2, ) * dims

    A = self.function_context["conv"](**conv_params)(layer_stack.pop())
    logger.debug(str(A))
    A = self.function_context["conv"](**conv_params)(A)
    logger.debug(str(A))
    B = self.function_context["pool"](**pool_params)(A)
    logger.debug(str(B))
    layer_stack.append(A)
    layer_stack.append(B)

  def latent_block(self, filter_size, layer_stack):
    """
    construct a latent space block
      conv
      conv
    or
      flatten
      dense
      reshape
    """
    # FIXME: if the latent step is not a dense layer, we should not do convolution here,
    #        just concatenate the outputs from the convolution encoders and pass them out
    conv_params = {}
    conv_params.update(self.encoder_conv_params)
    conv_params["filters"] = filter_size

    # FIXME: HACK
    conv_params["kernel_size"] = (3, ) * 2

    latent_params = {}
    latent_params.update(self.latent_params)
    latent_params["filters"] = filter_size

    if (self.function_context["latent"] is Convolution2D or
        self.function_context["latent_fn"] is Convolution3D):
      A = self.function_context["conv"](**conv_params)(layer_stack.pop())
      logger.debug(str(A))
      A = self.function_context["conv"](**conv_params, name="latent")(A)
      logger.debug(str(A))
    elif self.function_context["latent"] is Dense:
      latent_input = layer_stack.pop()
      logger.debug(str(latent_input))

      A = Flatten()(latent_input)
      logger.debug(str(A))

      A = function_context["latent"](**latent_params, name="latent")(A)
      logger.debug(str(A))
      A = function_context["latent"](units=tf.math.reduce_prod(latent_input.shape[1:]), activation=latent_params["activation"])(A)
      logger.debug(str(A))
      A = Reshape(target_shape=latent_input.shape[1:])(A)
      logger.debug(str(A))
    else:
      raise NotImplementedError("Unknown latent space mode: '%s'" % function_context["latent"])

    layer_stack.append(A)

  def decoder_block(self, filter_size, dims, layer_stack):
    """
    decoder block
      upsample/transpose convolution
      none/concatenate/add
      conv
      conv
    """
    conv_params = {}
    conv_params.update(self.encoder_conv_params)
    conv_params["filters"] = filter_size
    conv_params["kernel_size"] = (3, ) * dims

    pool_size = (2, ) * dims

    if (self.function_context["trns"] is UpSampling2D or
        self.function_context["trns"] is UpSampling3D):
      deconv = self.function_context["trns"](pool_size)(layer_stack.pop())
    elif (self.function_context["trns"] is Conv2DTranspose or
          self.function_context["trns"] is Conv3DTranspose):
      deconv = self.function_context["trns"](filter_size, stride_size=pool_size)(layer_stack.pop())

    logger.debug("linking: [%s]-[%s]" % (str(deconv), str(layer_stack[-1])))
    A = self.function_context["link"]([deconv, layer_stack.pop()])
    logger.debug(str(A))
    A = self.function_context["conv"](**conv_params)(A)
    logger.debug(str(A))
    A = self.function_context["conv"](**conv_params)(A)
    logger.debug(str(A))
    layer_stack.append(A)

  def output_block(self, type, filter_size, name, layer_stack):
    # FIXME: rename filter_size to shape, all the way
    self.output_conv_kernel = (1, ) * len(filter_size)

    if type == "segmentation":
      # segmentation prediction will produce one output per label
      return self.function_context["conv"](
               1,
               self.output_conv_kernel,
               activation="sigmoid",
               name=name)(layer_stack.pop())
    elif type == "image":
      # image prediction output will match the channel depth of the input
      return self.function_context["conv"](
               # self.channel_count, # FIXME: we don't have a global channel count anymore, match to an input? take as a param?
               3,
               self.output_conv_kernel,
               activation="sigmoid",
               name=name)(layer_stack.pop())
    elif type == "category":
      # size here is the number of categories in a one-hot encoding
      M = self.function_context["conv"](filter_size, self.output_conv_kernel, activation="relu")(layer_stack.pop())
      logger.debug(str(M))
      return self.function_context["global"]()(M)
    elif type == "numeric":
      # regression requires no activation function
      M = self.function_context["conv"](1, self.output_conv_kernel, activation="linear")(layer_stack.pop())
      logger.debug(str(M))
      return self.function_context["global"]()(layer_stack.pop())
    else:
      raise NotImplementedError("unknown output type: '%s'" % type)


  def __call__(self):
    """
    constructs a keras model representing this unet
    """
    layer_stack = []

    logger.info("UNET [%i] : %s" % (len(self.filter_sizes), str(self.filter_sizes)))

    if len(self.encoder_definitions) != 1:
      raise NotImplementedError("Cannot process multiple inputs with this architecture")

    # FIXME: take multiple inputs
    # FIXME: how does this affect the layer stack? They must all end up at the same
    #        shape for concat before the latent layer, and also be well-shaped for
    #        linking to the decoder layers (where linking is requested? should we
    #        just let the caller do whatever and pass the errors up? yes, I think so)
    for encoder_name, encoder in self.encoder_definitions.items():
      inputs = Input(encoder["shape"])
      layer_stack.append(inputs)
      logger.debug(str(inputs))

      # build encoder layers for all layers up the last layer
      for idx, filter_size in enumerate(self.filter_sizes[:-1]):
        name = "encoder_%i" % idx
        logger.info("building: '%s'" % name)
        self.encoder_block(filter_size, len(encoder["shape"])-1, layer_stack)

    # build a latent space layer with the last number of filters
    # FIXME: with multiple inputs, concatenate all the encoder outputs
    #        to a single large array before feeding to the latent layer
    latent_size = self.filter_sizes[-1]
    name = "latent"
    logger.info("building: '%s'" % name)
    self.latent_block(latent_size, layer_stack)

    # build decoder layers by reversing the filter_size array
    outputs = []

    for decoder_name, decoder in self.decoder_definitions.items():
      local_layer_stack = list(layer_stack)

      # only image or segmentation outputs require the full upscaling decoder
      if decoder["type"] == "image" or decoder["type"] == "segmentation":
        for i, filter_size in enumerate(self.filter_sizes[:-1][::-1]):
          idx = len(self.filter_sizes)+i
          name = "%s_%i" % (decoder_name, idx)
          logger.info("building: '%s'" % name)
          self.decoder_block(filter_size, len(decoder["size"]), local_layer_stack)

      # build the final output layer dependent on which data type is requested
      # FIXME: rename "size" to "shape", all the way
      output = self.output_block(decoder["type"], decoder["size"], decoder_name, local_layer_stack)
      logger.debug(str(output))
      outputs.append(output)

      # this is an error in the model structure, it is quite serious
      #assert len(local_layer_stack) == 0, "local_layer_stack not exhausted (%i)" % len(local_layer_stack)

    logger.info("created %i outputs" % len(outputs))

    model = Model(inputs=[inputs], outputs=outputs)

    return model
