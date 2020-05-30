#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Reference:
#   tensorflow tfrecord tutorial:
#       https://www.tensorflow.org/tutorials/load_data/tfrecord

import logging

import numpy as np
import tensorflow as tf


logger = logging.getLogger(__file__)


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _serialize_image_label_dataset(img, label):
    """Convert numpy array to example object."""
    if isinstance(img, type(tf.constant(0))):
        img = img.numpy()

    if img.dtype != np.uint8:
        raise TypeError(f'Provided image data has invalid data type: {img.dtype}')

    img_byte = img.reshape(-1, 1).tobytes()

    if isinstance(label, type(tf.constant(0))):
        label = label.numpy()

    label_byte = label.tobytes()

    feature = {
        'image': _bytes_feature(img_byte),
        'label': _bytes_feature(label_byte),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_to_tfrecord(imgs, labels, tfrecord_file):
    """Write image/label data to tfrecord file.

    Images and labels given to this are put into
    one tfrecord_file, so if needed to be batch,
    split data into small chunks beforehand.

    Args:
        imgs: 4-d numpy array
        labels: 1-d numpy array
        tfrecord_file: str
            tfrecord file path

    Returns:
        None
    """
    if len(imgs.shape) != 4 or imgs.shape[0] != labels.shape[0]:
        raise ValueError(
            'Input image and/or label shapes are invalid. '
            f'Image: {imgs.shape}, Label: {labels.shape}'
        )

    logger.info(f'Writing TFRecord file: {tfrecord_file}')

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(len(imgs)):
            example = _serialize_image_label_dataset(imgs[i], labels[i])
            writer.write(example)


def write_to_tfrecord_from_generator(gen, tfrecord_file, serializer=None):
    """Write data to tfrecord file.

    Args:
        gen: function which returns generator
        tfrecord_file: str
            tfrecord file path
        serializer: function to serialize data
            if not provided,
            serialize image and label as tf.float32 and tf.string,
            then labeled as 'image' and 'label' respectively

    Returns:
        None
    """
    feature_dataset = tf.data.Dataset.from_generator(
        gen, output_type=tf.string, output_shape=()
    )

    if serializer is not None:
        serialized_feature_dataset = feature_dataset.map(serializer)
    else:
        serialized_feature_dataset = feature_dataset

    writer = tf.data.experimental.TFRecordWriter(tfrecord_file)
    writer.write(serialized_feature_dataset)


_image_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
}


# used for test
def tf_serialize_example(img, label):
    """Convert tensor to string to be serializable."""
    tf_string = tf.py_function(
        _serialize_image_label_dataset,
        (img, label),
        tf.string)
    return tf.reshape(tf_string, ())


def _parse_image(example_proto):
    return tf.io.parse_single_example(example_proto, _image_feature)


def read_from_tfrecord(filepaths):
    dataset = tf.data.TFRecordDataset(filepaths)
    return dataset.map(_parse_image)
