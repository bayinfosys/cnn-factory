"""
serve a model via a REST API
"""
import os
import sys
import logging
import json
import yaml

from flask import Flask
from flask import request, jsonify
from flask.logging import default_handler
from flask_cors import CORS


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


model = None


@app.before_first_request
def default_model_loader():
  """
  load the model and prep for serving results

  NB: with this approach, we need a model "warm-up" phase on startup
  """
  global model

  os.environ.get("MODEL_PATH", "UNDEFINED")
  app.logger.info("loading model from '%s'" % MODEL_PATH)


def default_image_model_inference(data):
  """
  default image inference

  data: numpy array containing image data in shape(height, width, channels)
  """
  pass


#
# data reading functions
#

"""
data reader functions will take a request.data object
and return exactly the data required for the model to
process.
i.e., image data will create a numpy array, text data will
create strings, etc
"""
mime_readers = {
    "image/jpeg": lambda request: PIL.Image.open(request.data),
    "image/png": lambda request: PIL.Image.open(request.data),
    "application/json": lambda request: json.loads(request.data, strict=False),
    "application/yaml": lambda request: yaml.load(request.data, Loader=yaml.SafeLoader)
}

"""
perform model inference on data
"""
inference_functions = {
    "image/jpeg": default_image_model_inference,
    "image/png": default_image_model_inference,
}

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

  try:
    file_object = request.files["file"]
  except KeyError as e:
    return jsonify({"error": "no valid file content; expected a 'file' key"}), 400

  hashlib.md5(file_object.read(content_length)).hexdigest()

  try:
    data = mime_reader[content_type](request)
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
  try:
    results = inference_functions[content_type](data)
  except KeyError as e:
    return jsonify({"error": "unhandled inference-type: '%s'" % content_type}), 400
  except:
    app.logger.exception("could not process_upload")
    return jsonify({"error": "unknown error"}), 500

  # return the results
  return jsonify(results)


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


if __name__ == "__main__":
    app.run()
