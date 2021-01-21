import logging

from os import makedirs
from os.path import join

from common.schedulers import step_decay

from tensorflow.keras.callbacks import (TensorBoard as TensorBoardCallback,
                                        ModelCheckpoint as ModelCheckpointCallback,
                                        EarlyStopping as EarlyStoppingCallback,
                                        LearningRateScheduler)

#  from training.callbacks.callback_extns import TFRunMetaData
#  from training.callbacks.callback_extns import (WeightWriter,
#                                                 ValidationOutput)

logger = logging.getLogger(__name__)

# disable loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def create_keras_callbacks(
        model,
        modelname,
        output_dir,
    ):
  """
  creates callbacks for keras training process

  creates:
    + tensorboard callback: logs are written to <output_dir>/logs/<modelname>
    + checkpoint model: saves best validation score model to <output_dir>/<modelname>.hdf5
    + early stopping: stops training when validation hasn't improved for 25 epochs
    + learning rate schedular: reduces learning rate by half every 20 epochs

  returns:
    list of callbacks which can be passed directly to keras.fit

  TODO:
    TFRunMetaData: captures GPU usage, memory stats etc, but requires at least series 10 GPU
    WeightWriter: custom callback to output weights; needs integrating
    ValidationOutput: custom callback to output results of validation steps; needs integrating
  """
  tensorboard_dir = join(output_dir, "logs", modelname)
  logger.info("tensorboard logs writing to '%s'" % tensorboard_dir)

  checkpoint_filepath = join(output_dir, modelname + ".hdf5")
  logger.info("writing checkpoint file to '%s'" % checkpoint_filepath)

#  weight_dir = join(output_dir, modelname, "weights")
#  logger.info("weightwriter writing to '%s'" % weight_dir)

#  validation_dir = join(output_dir, model_name, "validation")
#  logger.info("validation data writing to '%s'" % validation_dir)

  for dir in [output_dir, tensorboard_dir]:
    try:
      makedirs(dir)
    except FileExistsError:
      logger.warning("'%s' already exists" % dir)
      continue
    except OSError:
      logger.exception("creating '%s' failed with OSError" % dir)
      continue

  tensorboard_callback = TensorBoardCallback(
      log_dir=tensorboard_dir,
      histogram_freq=0,
      write_graph=True,
      write_grads=False,
      write_images=False,
      embeddings_freq=0,
      embeddings_layer_names=None,
      embeddings_metadata=None)

  tensorboard_callback.set_model(model)

  return [
      tensorboard_callback,

      ModelCheckpointCallback(
          filepath=checkpoint_filepath,
          verbose=0,
          save_best_only=True
      ),

      EarlyStoppingCallback(
          monitor="val_loss",
          min_delta=0.00000001,
          patience=25,
          mode="min"
      ),

      LearningRateScheduler(step_decay(20, 0.5))

      #ReduceLROnPlateauCallback(
      #    monitor="val_loss",
      #    factor=0.5,
      #    patience=4,
      #    mode="min",
      #    epsilon=0.01,
      #    min_lr=0.0000001
      #),

      #TFRunMetaData(tensorboard_callback),

      #WeightWriter(weight_dir),

      #ValidationOutput(
      #    validation_dir,
      #    kwargs["validation_generator"],
      #    kwargs["validation_steps"]
      #)
  ]
