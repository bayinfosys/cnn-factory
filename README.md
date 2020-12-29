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
  anax32/cnn-factory \
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
