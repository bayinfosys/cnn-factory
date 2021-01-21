#!/bin/bash -u

##
#
# create a tensorboard visualisation of the logs dir
#
# run this and navigate to http://localhost:6006
#
##

docker run \
  --rm \
  -it \
  -u $(id -u):$(id -g) \
  -p 6006:6006 \
  -v $(pwd)/output/models/logs/:/tflogs \
  tensorflow/tensorflow \
    tensorboard \
      --logdir /tflogs \
      --bind_all \
      --port 6006
