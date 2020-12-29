"""
schedulers for learning rate
https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
"""

import logging


def step_decay(epoch_steps=20, reduction_amount = 0.5):
  """use a stepped decay rate.
  https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
  NB: use a higher momentum when dropping learning rates so the new
      learning rate doesn't fuck up the old weights
  https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
  NB: for keras >2.2.0 we need to accept a current_lr parameter
  """
  import logging

  logger = logging.getLogger(__name__)

  def f(epoch, lr=0.0001):
    if epoch > 0 and epoch % epoch_steps == 0:
      lr = lr * reduction_amount
      logger.info ("step_decay: updated lr to: %f" % lr)

    return lr

  return f

# FIXME: write a main entrypoint which outputs a bunch of data
#        illustrating the functions to a csv file.
#        take number of points as a cli arg
