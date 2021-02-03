import logging

from .models import get_model_memory_usage, ConvNet, UNet

logger = logging.getLogger(__name__)


MODEL_BUILDERS = {
  "convnet": ConvNet,
  "unet": UNet
}


def default_model_builder(
      model_type="unet",
      image_shape=(None, None),
      channel_count=3,
      network_depth=5,
      filter_sizes=None,
      output_definitions=None
    ):
  """
  build and return a model to the caller
  model_type: name of the model architecture type
  image_shape: spatial dimensions of image data
  channel_count: number of channels in the image
  network_depth: number of layers in the network
  outputs: map of output names to output definitions
           {<name>:{"size": <output shape>}}
  """
  logger.info("building model with: '%s', '%s', %i, %i" % (
      model_type, str(image_shape), channel_count, network_depth))

  # FIXME: filter_sizes is also a valid param
  try:
    model = MODEL_BUILDERS[model_type](
                image_shape=image_shape,
                channel_count=channel_count,
                network_depth=network_depth,
                filter_sizes=filter_sizes,
                output_definitions=output_definitions
            )()
  except KeyError as e:
    logger.error("'%s' is not a supported model type" % model_type)
    raise

  return model
