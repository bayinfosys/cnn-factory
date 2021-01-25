#!/bin/bash -u

##
#
# docker command to build a machine learning model
# using the cnn-factory docker image
#
# requires:
# + host dir containing images and labels mounted
# + host dir for writing out model data and tflogs
#
##

# run the host
docker run \
  -it \
  --rm \
  --runtime=nvidia \
  --read-only -v $(pwd)/data:/data \
  -v $(pwd)/output:/out \
  -u $(id -u):$(id -g) \
  bayis/cnn-factory \
    --help
