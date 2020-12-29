#!/bin/bash -u

docker run \
  -it \
  --rm \
  --runtime=nvidia \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/out \
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
