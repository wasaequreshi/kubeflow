# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX taxi model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Text, List

# At least one feature is needed.

# # Name of features which have continuous float values. These features will be
# # used as their own values.
# DENSE_FLOAT_FEATURE_KEYS = ['trip_miles']

# # Name of features which have continuous float values. These features will be
# # bucketized using `tft.bucketize`, and will be used as categorical features.
# BUCKET_FEATURE_KEYS = ['pickup_latitude']
# # Number of buckets used by tf.transform for encoding each feature. The length
# # of this list should be the same with BUCKET_FEATURE_KEYS.
# BUCKET_FEATURE_BUCKET_COUNT = [10]

# # Name of features which have categorical values which are mapped to integers.
# # These features will be used as categorical features.
# CATEGORICAL_FEATURE_KEYS = ['trip_start_hour']
# # Number of buckets to use integer numbers as categorical features. The length
# # of this list should be the same with CATEGORICAL_FEATURE_KEYS.
# CATEGORICAL_FEATURE_MAX_VALUES = [24]

# # Name of features which have string values and are mapped to integers.
# VOCAB_FEATURE_KEYS = []

# # Uncomment below to add more features to the model
# # DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']
# #
# # BUCKET_FEATURE_KEYS = [
# #     'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
# #     'dropoff_longitude'
# # ]
# # # Number of buckets used by tf.transform for encoding each feature.
# # BUCKET_FEATURE_BUCKET_COUNT = [10, 10, 10, 10]
# #
# # CATEGORICAL_FEATURE_KEYS = [
# #     'trip_start_hour', 'trip_start_day', 'trip_start_month'
# # ]
# # # Number of buckets to use integer numbers as categorical features.
# # CATEGORICAL_FEATURE_MAX_VALUES = [24, 31, 12]
# #
# #
# # VOCAB_FEATURE_KEYS = [
# #     'payment_type',
# #     'company',
# # ]

# # Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
# VOCAB_SIZE = 1000

# # Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
# OOV_SIZE = 10

# # Keys
# LABEL_KEY = 'big_tipper'


# def transformed_name(key: Text) -> Text:
#   """Generate the name of the transformed feature from original name."""
#   return key + '_xf'


# def vocabulary_name(key: Text) -> Text:
#   """Generate the name of the vocabulary feature from original name."""
#   return key + '_vocab'


# def transformed_names(keys: List[Text]) -> List[Text]:
#   """Transform multiple feature names at once."""
#   return [transformed_name(key) for key in keys]

"""Python source file includes Breast Cancer utils for Keras model.

This is used from Cifar10 example and modified for our use case.
"""


import os
from typing import List, Text
import absl
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import rewriter_factory
from tfx.dsl.io import fileio
from tfx_bsl.tfxio import dataset_options

import flatbuffers
# from tflite_support import metadata_schema_py_generated as _metadata_fb
# from tflite_support import metadata as _metadata

from tensorflow.keras.applications import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D,  LSTM, GlobalAveragePooling2D
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate

_TRAIN_DATA_SIZE = 128
_EVAL_DATA_SIZE = 128
_TRAIN_BATCH_SIZE = 32
_EVAL_BATCH_SIZE = 32
_CLASSIFIER_LEARNING_RATE = 1e-3
_FINETUNE_LEARNING_RATE = 7e-6
_CLASSIFIER_EPOCHS = 2

_IMAGE_KEY = 'image'
_LABEL_KEY = 'label'

_TFLITE_MODEL_NAME = 'tflite'


def _transformed_name(key):
  return key + '_xf'


def _get_serve_image_fn(model):
  """Returns a function that feeds the input tensor into the model."""

  @tf.function
  def serve_image_fn(image_tensor):
    """Returns the output to be used in the serving signature.

    Args:
      image_tensor: A tensor represeting input image. The image should have 3
        channels.

    Returns:
      The model's predicton on input image tensor
    """
    return model(image_tensor)

  return serve_image_fn


def _image_augmentation(image_features):
  """Perform image augmentation on batches of images .

  Args:
    image_features: a batch of image features

  Returns:
    The augmented image features
  """
  batch_size = tf.shape(image_features)[0]
  image_features = tf.image.random_flip_left_right(image_features)
  image_features = tf.image.resize_with_crop_or_pad(image_features, 250, 250)
  image_features = tf.image.random_crop(image_features,
                                        (batch_size, 224, 224, 3))
  return image_features


def _data_augmentation(feature_dict):
  """Perform data augmentation on batches of data.

  Args:
    feature_dict: a dict containing features of samples

  Returns:
    The feature dict with augmented features
  """
  image_features = feature_dict[_transformed_name(_IMAGE_KEY)]
  image_features = _image_augmentation(image_features)
  feature_dict[_transformed_name(_IMAGE_KEY)] = image_features
  return feature_dict


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              is_train: bool = False,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    is_train: Whether the input dataset is train split or not.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema)
  # Apply data augmentation. We have to do data augmentation here because
  # we need to apply data agumentation on-the-fly during training. If we put
  # it in Transform, it will only be applied once on the whole dataset, which
  # will lose the point of data augmentation.
  if is_train:
    dataset = dataset.map(lambda x, y: (_data_augmentation(x), y))

  return dataset


def _freeze_model_by_percentage(model: tf.keras.Model, percentage: float):
  """Freeze part of the model based on specified percentage.

  Args:
    model: The keras model need to be partially frozen
    percentage: the percentage of layers to freeze

  Raises:
    ValueError: Invalid values.
  """
  if percentage < 0 or percentage > 1:
    raise ValueError('Freeze percentage should between 0.0 and 1.0')

  if not model.trainable:
    raise ValueError(
        'The model is not trainable, please set model.trainable to True')

  num_layers = len(model.layers)
  num_layers_to_freeze = int(num_layers * percentage)
  for idx, layer in enumerate(model.layers):
    if idx < num_layers_to_freeze:
      layer.trainable = False
    else:
      layer.trainable = True


def _build_keras_model() -> tf.keras.Model:
  """Creates a Image classification model with MobileNet backbone.

  Returns:
    The image classifcation Keras Model and the backbone MobileNet model
  """
 
  base_model = VGG16(include_top=False, weights='imagenet',input_shape=(224, 224, 3), pooling='avg')
  base_model.input_spec = None

  model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(
        input_shape=(224, 224, 3), name=_transformed_name(_IMAGE_KEY)),
    base_model,
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2, activation='softmax')
  ])

  _freeze_model_by_percentage(base_model, 1.0)

  model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=_CLASSIFIER_LEARNING_RATE),
    metrics=['sparse_categorical_accuracy'])

  model.summary(print_fn=absl.logging.info)
  return model, base_model


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  # tf.io.decode_png function cannot be applied on a batch of data.
  # We have to use tf.map_fn
  image_features = tf.map_fn(
      lambda x: tf.io.decode_png(x[0], channels=3),
      inputs[_IMAGE_KEY],
      dtype=tf.uint8)
  # image_features = tf.cast(image_features, tf.float32)
  image_features = tf.image.resize(image_features, [224, 224])
  image_features = tf.keras.applications.mobilenet.preprocess_input(
      image_features)

  outputs[_transformed_name(_IMAGE_KEY)] = image_features
  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

  return outputs

# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.

  Raises:
    ValueError: if invalid inputs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=True,
      batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=False,
      batch_size=_EVAL_BATCH_SIZE)

  model, base_model = _build_keras_model()

  absl.logging.info('Tensorboard logging to {}'.format(fn_args.model_run_dir))
  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  # Our training regime has two phases: we first freeze the backbone and train
  # the newly added classifier only, then unfreeze part of the backbone and
  # fine-tune with classifier jointly.
  steps_per_epoch = int(_TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE)
  total_epochs = int(fn_args.train_steps / steps_per_epoch)
  if _CLASSIFIER_EPOCHS > total_epochs:
    raise ValueError('Classifier epochs is greater than the total epochs')

  absl.logging.info('Start training the top classifier')
  model.fit(
      train_dataset,
      epochs=_CLASSIFIER_EPOCHS,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  absl.logging.info('Start fine-tuning the model')
  # Unfreeze the top MobileNet layers and do joint fine-tuning
  _freeze_model_by_percentage(base_model, 0.9)

  # We need to recompile the model because layer properties have changed
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.RMSprop(lr=_FINETUNE_LEARNING_RATE),
      metrics=['sparse_categorical_accuracy'])
  model.summary(print_fn=absl.logging.info)

  model.fit(
      train_dataset,
      initial_epoch=_CLASSIFIER_EPOCHS,
      epochs=total_epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  # Prepare the TFLite model used for serving in MLKit
  signatures = {
      'serving_default':
          _get_serve_image_fn(model).get_concrete_function(
              tf.TensorSpec(
                  shape=[None, 224, 224, 3],
                  dtype=tf.float32,
                  name=_transformed_name(_IMAGE_KEY)))
  }
    #scs
  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  temp_saving_model_dir_original = os.path.join(fn_args.serving_model_dir, 'original')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)
  model.save(temp_saving_model_dir_original, save_format='tf', signatures=signatures)

  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER,
      name='tflite_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir, tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)

  # Add necessary TFLite metadata to the model in order to use it within MLKit
  # TODO(dzats@): Handle label map file path more properly, currently
  # hard-coded.
  tflite_model_path = os.path.join(fn_args.serving_model_dir,
                                   _TFLITE_MODEL_NAME)

  fileio.rmtree(temp_saving_model_dir)
