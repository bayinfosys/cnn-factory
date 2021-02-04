import logging

from .models import get_model_memory_usage, ConvNet, UNet

logger = logging.getLogger(__name__)


MODEL_BUILDERS = {
  "convnet": ConvNet,
  "unet": UNet
}


def default_model_builder(
      model_type="unet",
      network_depth=5,
      filter_sizes=None,
      input_definitions=None,
      output_definitions=None
    ):
  """
  build and return a model to the caller
  model_type: name of the model architecture type
  network_depth: number of layers in the network
  input_definitions: map of input names and definitions
  output_definitions: map of output names to output definitions
           {<name>:{"size": <output shape>}}
  """
  logger.info("building model with: '%s', %i" % (
      model_type, network_depth))

  # FIXME: filter_sizes is also a valid param
  try:
    model = MODEL_BUILDERS[model_type](
                network_depth=network_depth,
                filter_sizes=filter_sizes,
                input_definitions=input_definitions,
                output_definitions=output_definitions
            )()
  except KeyError as e:
    logger.error("'%s' is not a supported model type" % model_type)
    raise

  return model
