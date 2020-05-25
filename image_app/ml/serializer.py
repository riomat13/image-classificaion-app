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
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _serialize_data(img, label):
    img_byte = img.tobytes()
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
        tfrecord_file: str:
            tfrecord file path
    """
    if len(imgs.shape) != 4 or imgs.shape[0] != labels.shape[0]:
        raise ValueError(
            'Input image and/or label shapes are invalid. '
            f'Image: {imgs.shape}, Label: {labels.shape}'
        )

    logger.info(f'Writing TFRecord file: {tfrecord_file}')

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(len(imgs)):
            example = _serialize_data(imgs[i], labels[i])
            writer.write(example)


_image_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image(example_proto):
    return tf.io.parse_single_example(example_proto, _image_feature)


def read_from_tfrecord(filepaths):
    dataset = tf.data.TFRecordDataset(filepaths)
    return dataset.map(_parse_image)
