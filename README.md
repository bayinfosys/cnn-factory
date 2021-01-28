# CNN-factory

docker image to train CNN style neural networks

## Usage

+ mount data from host into `/data` directory
+ pass a `glob` of images and masks to the training script
+ train
+ get the model from the mounted `/out/models` directory

```bash
docker run \
  -it \
  --rm \
  --runtime=nvidia \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/out \
  bayis/cnn-factory \
    --modelname 'my-model' \
    --images '/data/images/*.png' \
    --masks '/data/labels/*.png' \
    --batch-size 1 \
    --num-augs 4 \
    --num-epochs 800 \
    --output-path '/out/models/' \
    --shuffle-data \
    --learning-rate 0.0001
```

## ARGS file

A json file with arg values can be passed into the container:

```bash
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
  bayis/cnn-factory
```
