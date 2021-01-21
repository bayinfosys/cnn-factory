# Image Resize

Container to transform a set of images from one size to another

## Usage

```bash
docker build -t bayis/transformer . && docker run -it --rm bayis/transformer
```

with a directory `./data/images` of image data of all different sizes, we transform
those images to `./data/images-256-256` with:

```bash
docker run \
  -it \
  --rm \
  -e LOG_LEVEL=INFO \
  -v $(pwd):/data \
  bayis/transformer \
  -i /data/images/*.jpg \
  -o /data/images-256-256 \
  -s 256 256
```
