#!/bin/bash -u

##
#
# docker command to mount the src dir for local development
#
##

# create dummy data
mkdir -p ./data/dummy/images
mkdir -p ./data/dummy/labels

for i in {001..010}; do
  touch ./data/dummy/images/$i.png
  touch ./data/dummy/labels/$i.png
done

# run the host
docker run \
  -it \
  --rm \
  --runtime=nvidia \
  -v $(pwd)/src:/src \
  -v $(pwd)/examples/args:/args \
  -v $(pwd)/data/dummy:/data \
  -v $(pwd)/output:/out \
  -u $(id -u):$(id -g) \
  -e ARGS_FILE=/args/train.args.run.01 \
  anax32/cnn-factory
