import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import (Input, Activation,
                                     Convolution2D, AveragePooling2D, MaxPooling2D,
                                     UpSampling2D, Conv2DTranspose,
                                     Convolution3D, MaxPooling3D,
                                     UpSampling3D, Conv3DTranspose,
                                     Dropout)

from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras import backend as K

def get_model_memory_usage (batch_size, model):
  """estimate the memory usage of the model.
  NB: this estimates the number of bytes required
  by the variables; the actual usuage will be
  higher because of the backprop variables and
  storage of activations, etc."""
  shapes_mem_count = 0

  for l in model.layers:
    single_layer_mem = 1

    layer_output_shape = l.output_shape if isinstance(l.output_shape, list) else [l.output_shape]

    for shape in [x for x in layer_output_shape if x is not None]:
      if shape is None:
        continue

      for dim in [x for x in shape if x is not None]:
        single_layer_mem *= dim
        shapes_mem_count += single_layer_mem

  # FIXME: these used to be set() of model.trainable_weights, but had to be removed
  #        because of a tensorflow error since tensorflow 2.0, check the issues at some point
  trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
  non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))

  return 4*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)

def get_vgg_network (image_shape):
  base_model = VGG16 (include_top=False, input_shape = (image_shape[0], image_shape[1], 3))

  Conv2D = Convolution2D
  
  # create the new top
  LD = {}   # layer dict
  [LD.update({l.name: l}) for l in base_model.layers]
  
  # unpool block 5
  block5_pool = base_model.output #LD["block5_pool"]
  unblock5_unpool = Conv2DTranspose (512, (2,2), strides=(2,2)) (base_model.output)
  unblock5_conv3 = Conv2D (512, (3,3), padding="same") (unblock5_unpool)#(LD["block5_conv3"].output)
  unblock5_conv2 = Conv2D (512, (3,3), padding="same") (unblock5_conv3)#(LD["block5_conv2"].output)
  unblock5_conv1 = Conv2D (512, (3,3), padding="same") (unblock5_conv2)#(LD["block5_conv1"].output)
  
  # next unpool block
  #block4_pool = LD["block4_pool"]
  unblock4_unpool = Conv2DTranspose (256, (2,2), strides=(2,2)) (unblock5_conv1) #(block4_pool.output)
  unblock4_conv3 = Conv2D (256, (3,3), padding="same") (unblock4_unpool)#(LD["block4_conv3"].output)
  unblock4_conv2 = Conv2D (256, (3,3), padding="same") (unblock4_conv3)#(LD["block4_conv2"].output)
  unblock4_conv1 = Conv2D (256, (3,3), padding="same") (unblock4_conv2)#(LD["block4_conv1"].output)
  
  #block3_pool = LD["block3_pool"]
  unblock3_unpool = Conv2DTranspose (128, (2,2), strides=(2,2)) (unblock4_conv1) #(block3_pool.output)
  unblock3_conv3 = Conv2D (128, (3,3), padding="same") (unblock3_unpool) #(LD["block3_conv3"].output)
  unblock3_conv2 = Conv2D (128, (3,3), padding="same") (unblock3_conv3) #(LD["block3_conv2"].output)
  unblock3_conv1 = Conv2D (128, (3,3), padding="same") (unblock3_conv2) #(LD["block3_conv1"].output)
  
  #block2_pool = LD["block2_pool"]
  unblock2_unpool = Conv2DTranspose (64, (2,2), strides=(2,2)) (unblock3_conv1) #(block2_pool.output)
  unblock2_conv2 = Conv2D (64, (3,3), padding="same") (unblock2_unpool) #(LD["block2_conv2"].output)
  unblock2_conv1 = Conv2D (64, (3,3), padding="same") (unblock2_conv2) #(LD["block2_conv1"].output)
  
  #block1_pool = LD["block1_pool"]
  unblock1_unpool = Conv2DTranspose (32, (2,2), strides=(2,2)) (unblock2_conv1)#(block1_pool.output)
  unblock1_conv2 = Conv2D (32, (3,3), padding="same") (unblock1_unpool) #(LD["block1_conv2"].output)
  unblock1_conv1 = Conv2D (32, (3,3), padding="same") (unblock1_conv2) #(LD["block1_conv1"].output)
  
  output = Conv2D (1, (1,1), activation="softmax") (unblock1_conv1)
  
  model = Model (inputs=base_model.input, outputs=output)
  
  for layer in base_model.layers:
    layer.trainable = False
    
  return model

def get_unet_us_untouched (image_shape):

  with tf.device ("/gpu:0"):
    inputs = Input ((image_shape[0], image_shape[1], image_shape[2], 1)) # tensor-flow

    conv1 = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Convolution3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Convolution3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Convolution3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Convolution3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

#  with tf.device ("/gpu:1"):
    conv5 = Convolution3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Convolution3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
    conv6 = Convolution3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Convolution3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
    conv7 = Convolution3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Convolution3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
    conv8 = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    conv9 = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Convolution3D(1, (1, 1, 1), activation='sigmoid')(conv9)

  model = Model(inputs=[inputs], outputs=[conv10])

#  model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

  return model
