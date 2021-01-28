#/bin/bash -u

##
#
# build the docker images
#
# NB: run from the project root directory, i.e.:
# > ./scripts/build-images.sh
#
##

docker \
  build \
  -t bayis/generator \
  -f docker/generator/Dockerfile \
  .

docker \
  build \
  -t bayis/cnn-factory \
  -f docker/train/Dockerfile \
  .

docker \
  build \
  -t bayis/cnn-serve \
  -f docker/serve/Dockerfile \
  .
