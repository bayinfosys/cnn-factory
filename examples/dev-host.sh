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
  -v $(pwd)/data/dummy:/data \
  -v $(pwd)/output:/out \
  -u $(id -u):$(id -g) \
  anax32/cnn-factory \
  python3 /src/train.py \
    --modelname 'my-model' \
    --images '/data/images/*.png' \
    --masks '/data/labels/*.png' \
    --batch-size 1 \
    --num-augs 4 \
    --num-epochs 800 \
    --output-path '/out/models/' \
    --shuffle-data \
    --learning-rate 0.0001
