# Image generators

Functions which produce generators which yield image data to the caller.

The generators do not load nor preprocess the images: this functionality
is handled by a `image_preprocess_fn` and `label_preprocess_fn` which is
passed to the generator creating function.

The generators are responsible for shuffling the input lists, checking the
return values of the `*_preprocess_fn` and packing the results into
correctly shaped and typed numpy arrays.

They can also produce debug images to investigate the actual image/label
data received by the training/validation process.

Image augmentation is also handled inside the generator via a callback hook
and can be investigated with debug outputs.
