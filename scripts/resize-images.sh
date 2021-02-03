#! /bin/bash -u

##
#
# convert a directory of images to 128-128 size
# format is preserved
##

docker run \
  -it \
  --rm \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/data:/data:ro \
  -v $(pwd)/data:/output \
  bayis/transformer \
    -i /data/images/*.jpg \
    -o /output/images-128-128 \
    -s 128 128
