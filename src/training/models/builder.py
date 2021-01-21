from .misc import get_model_memory_usage
from .convnet import ConvNet
from .convnet_with_labels import ConvNetWithLabels
from .unet import UNet
from .unet_with_labels import UNetWithLabels

MODEL_BUILDERS = {
  "convnet": ConvNet,
  "unet": UNet
}


def default_model_builder(
      model_type="unet",
      image_shape=(None, None),
      channel_count=3,
      network_depth=5
    ):
  """
  build and return a model to the caller
  """
  logger.info("building model with: '%s', '%s', %i, %i" % (
      model_type, str(image_shape), channel_count, network_depth))

  # FIXME: filter_sizes is also a valid param
  try:
    model = MODEL_BUILDERS[model_type](
                image_shape=image_shape,
                channel_count=channel_count,
                network_depth=network_depth
            )()
  except KeyError as e:
    logger.error("'%s' is not a supported model type" % model_type)
    raise

  return model
