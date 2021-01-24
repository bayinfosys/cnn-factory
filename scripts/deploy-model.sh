#!/bin/bash -u

##
#
# docker command to deploy a machine learning model
# with a rest api using the cnn-serve docker image
#
# requires:
# + host dir for writing out model data and tflogs
# + environment variable with path to model file
#
##

MODEL_PATH=${1:-"/models/model.hdf5"}

# run the host
docker run \
  -it \
  --rm \
  --runtime=nvidia \
  -v $(pwd)/output/models:/models \
  -u $(id -u):$(id -g) \
  -p 8080:5000 \
  -e MODEL_PATH=${MODEL_PATH} \
  -e MODEL_TYPE="keras-common-loss" \
  -e LOG_LEVEL=DEBUG \
  bayis/cnn-serve
