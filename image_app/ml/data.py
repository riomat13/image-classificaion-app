#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import glob
import math
import random
from functools import partial

import numpy as np
import tensorflow as tf

from image_app.settings import ROOT_DIR
from image_app.ml.base import LabelData
from image_app.ml.preprocess import (
    reformat_image,
    load_image,
    preprocess_input
)
from image_app.ml.serializer import read_from_tfrecord


logger = logging.getLogger(__file__)


def convert_image_to_tensor(image):  # pragma: no cover
    """Convert single image to tensor."""
    x = reformat_image(image, target_size=(224, 224))
    x = preprocess_input(x)
    return x


def convert_images_to_tensor(images):
    """Convert PIL images to tensor for MobileNet.
    This is used for directly loading data from file without `tf.data`.

    Args:
        images: PIL.Image.Image or list of PIL.Image.Image

    Returns:
        numpy.ndarray, shape = (n_images, 224, 224, 3)
    """
    x = np.zeros((len(images), 224, 224, 3), dtype=np.float32)

    for i, img in enumerate(images):
        x[i, ...] = convert_image_to_tensor(img)

    return x


def get_label_from_path(filepath):
    """Get label based on filepath.

    (This may be updated later.)
    """
    labels = LabelData.get_label_data()
    label = os.path.basename(os.path.dirname(filepath))
    label = label.replace('_', ' ').capitalize()
    return labels[label]


def _load_images(files):
    """Load images and return the image data and the label sets.
    If failed to load an image, it would be skipped,
    and return without it.
    All images will be resized to (224, 224, 3).
    """
    # to handle exception during reading file
    imgs = np.zeros((len(files), 224, 224, 3), np.uint8)
    labels = np.zeros((len(files),), np.uint8)

    idx = 0
    for f in files:
        try:
            imgs[idx, ...] = load_image(f, target_size=(224, 224), dtype=np.uint8)
            labels[idx] = get_label_from_path(f)
            idx += 1
        except Exception as e:
            logger.warning(f'Failed to load: {f} ({e})')

    # truncate failed files
    if idx < len(imgs):
        imgs = imgs[:idx]
        labels = labels[:idx]

    return imgs, labels


def extract_image_file_paths(dirpath, depth=0):  # pragma: no cover
    """Read images inside directory.
    Extention of the images must be `.jpg`.

    Args:
        dirpath: str
            path to directory
        depth: int
            depth to the image files from the given dirpath
            if depth is 0, `${dirpath}/*.jpg`,
            if depth is 1, `${dirpath|/*/*.jpg`.

    Returns:
        list of files with '.jpg' extention

    Raises:
        ValueError: if depth < 0
    """
    if depth < 0:
        raise ValueError('depth must be 0 or positive')

    # add 'png' and 'jpeg'?
    return [
        f for f in glob.glob(os.path.join(dirpath, '/'.join(['*'] * depth), '*.jpg'))
    ]


def load_images_all(files, shuffle=True):
    """Read images inside directory.
    Extention of the images must be `.jpg`.

    Args:
        files: a list of str or path like object
            image paths
        shuffle: bool (default: True)
            shuffle given files if set to True

    Returns:
        tuple of numpy array: (images, labels)
            images: numpy.ndarray (np.uint8): shape (num_image, 224, 224, 3)
            labels: numpy.ndarray (np.uint8): shape (num_image,)
    """
    if shuffle:
        random.shuffle(files)

    return _load_images(files)


def load_images_all_by_batch(files, shuffle=True, batch_size=128):
    """Load image and return them as numpy array with given batch size
    instead of returning all data once.

    Extention of the images must be `.jpg`.

    Args:
        files: a list of str or path like object
            image paths
        shuffle: bool (default: True)
            shuffle given files if set to True
        batch_size: int (default: 128)
            batch size of output

    Returns:
        tuple of numpy array: (images, labels) with batch_size
            images: numpy.ndarray (np.uint8): shape (batch_size, 224, 224, 3)
            labels: numpy.ndarray (np.uint8): shape (batch_size,)

    Raises:
        ValueError: if batch_size < 0
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive. To fetch all data once, set batch_size large or use `load_images` instead')

    if shuffle:
        random.shuffle(files)

    for idx in range(0, len(files), batch_size):
        yield _load_images(files[idx:idx+batch_size])


def _single_img_data_augmentation(img,
                                  label,
                                  lr_flip=False,
                                  ud_flip=False,
                                  brightness=None,
                                  central_crop=None,
                                  saturation=None,
                                  random_brightness=None,
                                  chains_only=False,
                                  chains=[],
                                  probability=1.0):  # pragma: no cover
    """Image augmentation with `tf.image`."""
    if not chains_only:
        if lr_flip and random.random() < probability:
            new_img = tf.image.flip_left_right(img)
            yield new_img, label
        if central_crop and random.random() < probability:
            new_img = tf.image.resize(tf.image.central_crop(img, central_crop), (224, 224))
            yield new_img, label
        if ud_flip and random.random() < probability:
            new_img = tf.image.flip_up_down(img)
            yield new_img, label
        if brightness and random.random() < probability:
            new_img = tf.image.adjust_brightness(img, brightness)
            yield new_img, label
        if random_brightness and random.random() < probability:
            new_img = tf.image.random_brightness(img, random_brightness)
            yield new_img, label
        if saturation and random.random() < probability:
            new_img = tf.image.adjust_saturation(img, saturation)
            yield new_img, label

    for chain in chains:
        if random.random() > probability:
            continue

        new_img = tf.identity(img)
        for key in chain:
            if key not in _single_img_data_augmentation.__code__.co_varnames:
                logger.warn(f'Step [{key}] is not registered or not provided as argument. Skipping this.')
            if key == 'lr_flip':
                new_img = tf.image.flip_left_right(new_img)
            elif key == 'central_crop':
                new_img = tf.image.resize(tf.image.central_crop(new_img, central_crop), (224, 224))
            elif key == 'ud_flip':
                new_img = tf.image.flip_up_down(new_img)
            elif key == 'brightness':
                new_img = tf.image.adjust_brightness(new_img, brightness)
            elif key == 'random_brightness':
                new_img = tf.image.random_brightness(img, random_brightness)
            elif key == 'saturation':
                new_img = tf.image.adjust_saturation(new_img, saturation)
        yield new_img, label


def data_augmentation(imgs,
                      labels,
                      lr_flip=False,
                      ud_flip=False,
                      brightness=None,
                      saturation=None,
                      random_brightness=None,
                      chains_only=False,
                      chains=[],
                      probability=1.0):  # pragma: no cover
    """Image augmentation with `tf.image`.
    Generate image as generator.

    Args:
        imgs: tensor or numpy array of images
        labels: tensor or numpy array of labels corresponding to imgs
        lr_flip: bool (defailt: False)
            set to True to flip horizontally
        up_flip: bool (default: False)
            set to True to flip vertically
        brightness: float
            adjust brightness. Add the given value in entire image
        saturation: float
            factor to multiply the saturation by
            similar to brightness but before adding value,
            convert RGB to HSV, then add the value and convert back to RGB
            (use `tf.image.adjust_saturation`)
        probability: float
            probability to execute image augmentation,
            1.0 for applying all process to all images

    Returns:
        generator of a tuple of 3-d numpy arraya and 1-d numpy array
    """
    for img, label in zip(imgs, labels):
        yield from _single_img_data_augmentation(
            img, label,
            lr_flip=lr_flip,
            ud_flip=ud_flip,
            brightness=brightness,
            saturation=saturation,
            random_brightness=random_brightness,
            chains_only=chains_only,
            chains=chains
        )


def train_test_split(files, test_ratio):
    """Split files into train set and test set.

    Args:
        files: a list of path like object
        test_ratio: float
            ratio of dataset used for test

    Returns:
        a tuple of image and label tuple
        (train_imgs, train_labels), (test_imgs, test_labels)
    """
    random.shuffle(files)

    size = int(len(files) * test_ratio)

    train = files[size:]
    test = files[:size]

    return train, test


def load_tfrecord_files(dirpath, filename='*.tfrecord'):  # pragma: no cover
    files = glob.glob(os.path.join(dirpath, filename))
    dataset = read_from_tfrecord(files)
    return dataset


def data_generator(filepaths, buffer_size=6000, shuffle=True, repeat=False, batch_size=32):
    """Generate data batch from tfrecord file.

    Args:
        filepaths: list or tuple
            list of filepaths to read
        buffer_size: int (default: 6000)
            buffer size to be used for shuffling
            default is 6000 which can cover all training data.
        shuffle: bool (default: True)
            set to True, if need to shuffle
        repeat: int (default: 1)
            number of iteration for each tfrecord file to load
        batch_size: int (default: 32)
            batch size to generate data in each step

    Returns:
        generator of (image, label) dataset
            image will be processed for MobileNet
            shape: image - (batch_size, 224, 224, 3)
                   label - (batch_size, 1)
    """
    dataset = read_from_tfrecord(filepaths)

    if shuffle:  # pragma: no cover
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    if repeat:  # pragma: no cover
        dataset = dataset.repeat()

    for data in dataset.batch(batch_size):
        # preprocess(standardize) beforehand is 10-15% slower
        # and cosume much more disk space
        imgs = tf.io.decode_raw(data['image'], out_type=tf.uint8)
        imgs = tf.cast(tf.reshape(imgs, shape=(-1, 224, 224, 3)), dtype=tf.float32)
        imgs = preprocess_input(imgs)
        labels = tf.io.decode_raw(data['label'], out_type=tf.uint8)

        yield imgs, tf.squeeze(labels)
