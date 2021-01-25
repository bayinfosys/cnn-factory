"""
serve a model via a REST API
"""
import io
import os
import sys
import logging
import json
import yaml

import hashlib

from flask import Flask
from flask import request, jsonify, send_file
from flask.logging import default_handler
from flask_cors import CORS

import PIL
import numpy as np

app = Flask(__name__)
CORS(app)

# setup logging
# FIXME: move to logging config file
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "[%(name)s - %(filename)s:%(lineno)s:%(funcName)s] %(levelname)s - %(message)s"
    )
)
app.logger.addHandler(handler)
app.logger.removeHandler(default_handler)
# app.logger.setLevel(logging.DEBUG)
app.logger.setLevel(os.getenv("LOG_LEVEL", "WARN"))


# model global var
model = None

#
# MODEL LOADING
#
def keras_model_loader(filename):
  from tensorflow.keras.models import load_model
  return load_model(filename)


def keras_model_loader_with_common_loss(filename):
  """
  keras model loader which imports the common.loss functions
  used for semantic segmentation
  FIXME: can't we serialise the model without this information?
  """
  from tensorflow.keras.models import load_model
  from .common.loss import (dice_coef,
                            dice_coef_loss,
                            binary_cross_entropy_plus_dice_loss as bce_dice,
                            jaccard_index,
                            jaccard_index_loss)

  custom_loss_fns = {
    "dice_coef": dice_coef,
    "dice_coef_loss": dice_coef_loss,
    "binary_cross_entropy_plus_dice_loss": bce_dice,
    "jaccard_index": jaccard_index,
    "jaccard_index_loss": jaccard_index_loss
  }

  return load_model(filename, custom_objects=custom_loss_fns)


"""
load models from specific libraries
"""
model_loaders = {
  "keras": keras_model_loader,
  "keras-common-loss": keras_model_loader_with_common_loss,
#  "gensim": gensim_model_loader,
}


@app.before_first_request
def default_model_loader():
  """
  load the model and prep for serving results

  NB: with this approach, we need a model "warm-up" phase on startup

  FIXME: this method does not send the 500 responses
  """
  global model

  model_type = os.getenv("MODEL_TYPE", "UNDEFINED")
  model_path = os.getenv("MODEL_PATH", "UNDEFINED")
  app.logger.info("MODEL_TYPE: '%s', MODEL_PATH: '%s'" % (model_type, model_path))

  try:
    model = model_loaders[model_type](model_path)
  except KeyError as e:
    app.logger.error("unknown model type: '%s' [%s]" % (model_type, str(e)))
    model = None
    return jsonify({"error": "unable to initialise"}), 500
  except Exception as e:
    app.logger.exception(e)
    model = None
    return jsonify({"error": "unable to initialise"}), 500
  finally:
    app.logger.info("model: '%s'" % (str(model)))


#
# MODEL INFERENCE
#
def keras_inference(data):
  """
  infer with a keras model
  """
  app.logger.debug("infering on: '%s:%s' [%0.2f -> %0.2f]" % (
      str(data.shape), str(data.dtype), data.min(), data.max()))
  p = model.predict(data, batch_size=1, verbose=0)
  app.logger.debug("inference: '%s:%s' [%0.2f -> %0.2f]" % (
      str(p.shape), str(p.dtype), p.min(), p.max()))
  p_ = (p[0, ..., 0]*255.0).astype(np.uint8)
  app.logger.debug("inference: '%s:%s' [%0.2f -> %0.2f]" % (
      str(p_.shape), str(p_.dtype), p_.min(), p_.max()))

  img = PIL.Image.fromarray(p_, 'L')

  output = io.BytesIO()
  img.save(output, format="png")
  output.seek(0)
  return output, "image/png"


model_inference_functions = {
  "keras": keras_inference,
  "keras-common-loss": keras_inference
}

#
# DATA READER
#
"""
data reader functions will take a request.data object
and return exactly the data required for the model to
process.
i.e., image data will create a numpy array, text data will create strings, etc

all model types which work on particular data must have
a model_inference_fn which accepts exactly this data as
input, and performs any transformation required by the
library internally.

SPECIALISATION ORDER OF PRECEDENCE is DATATYPE -> MODELTYPE -> LIBRARY
"""
mime_readers = {
    "image/jpeg": lambda buffer: np.asarray(PIL.Image.open(buffer))[np.newaxis, ...],
    "image/png": lambda buffer: np.asarray(PIL.Image.open(buffer))[np.newaxis, ...],
    "application/json": lambda buffer: json.loads(buffer, strict=False),
    "application/yaml": lambda buffer: yaml.load(buffer, Loader=yaml.SafeLoader)
}

#
# data reading functions
#
@app.route("/infer", methods=["POST", "PUT"])
def inference():
  """
  post data for inference

  this function handles all the verification of the upload
  format, extraction of data in model compatible form, and
  passing to 'process_upload'.
  """
  try:
    content_type = request.headers["Content-Type"]
  except KeyError as e:
    app.logger.error("KeyError looking for content type: '%s'" % str(e))
    return jsonify({"error": "require content-type header"}), 400

  try:
    data_length = int(request.headers.get("Content-Length"))
  except KeyError:
    return jsonify({"error": "require content-length header"}), 411

  bytes_buffer = io.BytesIO(request.data)

  data_hash = hashlib.md5(bytes_buffer.read()).hexdigest()
  bytes_buffer.seek(0)
  app.logger.info("data_hash: '%s'" % data_hash)

  try:
    data = mime_readers[content_type](bytes_buffer)
  except KeyError as e:
    # FIXME: use flask-accept or some other library to negotiate content type
    return jsonify({"error": "unhandled content-type: '%s'" % content_type}), 400
  except (json.decoder.JSONDecodeError, yaml.YAMLError) as e:
    app.logger.error("parse error: '%s'" % str(e))
    return jsonify({"error": "could not parse data: %s" % str(e)}), 400
  except PIL.UnidentifiedImageError as e:
    app.logger.error("PIL.UnidentifiedImageError: '%s'" % str(e))
    return jsonify({"error": "could not process image data; unknown format"}), 400

  # run the model against the data file
  model_type = os.getenv("MODEL_TYPE", "UNDEFINED")
  app.logger.info("MODEL_TYPE: '%s'" % model_type)

  try:
    inference, mimetype = model_inference_functions[model_type](data)
  except KeyError as e:
    return jsonify({"error": "unhandled inference-type: '%s'" % model_type}), 400
  except:
    app.logger.exception("could not process_upload")
    return jsonify({"error": "unknown error"}), 500

  # return the results
  return send_file(inference, mimetype=mimetype)


@app.route("/status", methods=["GET"])
def status():
  """
  get the status of this instance
  """
  return jsonify(success=True)


@app.route("/version", methods=["GET"])
def version():
  """
  return the version of this container
  """
  return jsonify({"version": "0.2"})


#
# aws sagemaker api
#   https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html
#
@app.route("/invocations", methods=["POST"])
def aws_sagemaker_invocation():
  pass


@app.route("/ping", methods=["GET"])
def aws_sagemaker_ping():
  return "hi lol", 200


if __name__ == "__main__":
    app.run()
