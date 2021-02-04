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
    --csv input-data.csv \
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

# CSV file

Input data is specified via a CSV file.

+ CSV file must have column names in the first row.
+ Column headers are used to associate csv data to network inputs/outputs.
+ Input names and output names must not overlap.

## Tips

if you have a directory of images and labels as:
```bash
.
|-- ./images/*.jpg
|-- ./labels/*.png
```

create a csv file quickly in bash with simple commands:
```bash
paste -d <(ls -1 ./images/ | sed 's/^/images\/g') <(ls -1 ./labels | sed (s/^/labels\/g) > data.csv
```
NB: `ls` does not produce relative filepaths by default.

## Container mounts

You must add a prefix to the file name which is appropriate to the docker container mount point of the
data directory. i.e., to mount the above directory under `/data/` in the container, then use
`/data/images/` and `/data/labels/` as directory prefix for the image filenames and label filenames.

# Output definitions

network outputs are defined with a `json` formatted parameter.


```json
{
  "mask": {
    "type": "segmentation",
    "loss": "binary_crossentropy",
    "size": [128, 128],
    "weight": 1.0,
    "label": 1
  },
  "classification": {
    "type": "category",
    "loss": "categorical_crossentropy",
    "size": 56,
    "weight": 1.4,
  }
}
```

In this example, `mask` is a segmentation image, and `classification` is a categorical variable.
The location of the mask file will be taken from the csv file column `mask`.
The value for the `classification` field will be taken from the csv file column `classification`.

Internally, the `cnn-factory` will generate decoders for each of the outputs.
The `mask` output will produce a single channel image output;
`classification` will produce a one-hot encoded vector with 56 dimensions.

## Note on segmentations outputs

If you have a label map with multiple labels, setup multiple outputs and use the `label` field
to indicate which values from the label map will be used for that decoder output. The generator
will select the labels in the labelmap and merge them into a single boolean mask.
