#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import math
import random
from functools import partial

import numpy as np
import tensorflow as tf

from image_app.settings import ROOT_DIR
from image_app.ml.base import LabelData
from image_app.ml.serializer import write_to_tfrecord, read_from_tfrecord


def convert_image_to_tensor(image):  # pragma: no cover
    """Convert single image to tensor for MobileNet."""
    x = tf.keras.preprocessing.image.img_to_array(image)
    x = tf.keras.applications.mobilenet.preprocess_input(x)
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
    failed = 0

    for i, img in enumerate(images):
        x[i, ...] = convert_image_to_tensor(img)

    return x


def get_label_from_path(filepath):
    """Get label based on filepath.

    This may be updated later.
    """
    labels = LabelData.get_label_data()
    label = os.path.basename(os.path.dirname(filepath))
    label = label.replace('_', ' ').capitalize()
    return labels[label]


def load_image(filepath):  # pragma: no cover
    """Read an image file."""
    return tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))


def load_image_to_tensor(filepath):  # pragma: no cover
    img = load_image(filepath)
    return convert_image_to_tensor(img), get_label_from_path(filepath)


def load_images_all(dirpath, depth=0):
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
        tuple of numpy array: (images, labels)
            images: numpy.ndarray (np.uint8): shape (num_image, 224, 224, 3)
            labels: numpy.ndarray (np.uint8): shape (num_image,)

    Raises:
        ValueError: if depth < 0
    """
    if depth < 0:
        raise ValueError('depth must be 0 or positive')

    # add 'png' and 'jpeg'?
    files = [
        f for f in glob.glob(os.path.join(dirpath, '/'.join(['*'] * depth), '*.jpg'))
    ]

    # to handle exception during reading file
    failes = 0
    imgs = np.zeros((len(files), 224, 224, 3), np.float32)
    labels = np.zeros((len(files),), np.uint8)

    for i, f in enumerate(files):
        try:
            img = tf.keras.preprocessing.image.load_img(f, target_size=(224, 224))
            imgs[i, ...] = img
            labels[i] = get_label_from_path(f)
        except Exception as e:
            failes += 1

    if failes:
        imgs = imgs[:-failes]
        labels = labels[:-failes]

    imgs = imgs.astype(np.uint8)

    return imgs, labels


def _data_augmentation(imgs,
                       labels,
                       ratio,
                       flip=False,
                       central_crop=None,
                       saturation=None):  # pragma: no cover
    """Image augmentation with `tf.image`.

    Args:
        imgs: tensor or numpy array of images
        labels: tensor or numpy array of labels corresponding to imgs
        ratio: float
            ratio of data to be used for augmentation
            if set to 1.0, use all given data
            take images from first since they are already shuffled

    Returns:
        a tuple of augmented imgs and labels tensor
    """
    use_data_size = int(imgs.shape[0] * ratio)

    new_imgs = []
    new_labels = []

    for img, label in zip(imgs[:use_data_size], labels[:use_data_size]):
        if flip:
            new_imgs.append(tf.cast(tf.image.flip_left_right(img), tf.uint8))
            new_labels.append(label)
        if central_crop:
            new_img = tf.image.resize(tf.image.central_crop(img, central_crop), (224, 224))
            new_imgs.append(tf.cast(new_img, tf.uint8))
            new_labels.append(label)
        if saturation:
            new_img = tf.image.adjust_saturation(img, saturation)
            new_imgs.append(tf.cast(new_img, tf.uint8))
            new_labels.append(label)

    return tf.stack(new_imgs).numpy(), np.array(new_labels)


def train_test_split(imgs, labels, test_ratio):
    """Split image and label dataset into train and test sets.

    Args:
        imgs: numpy array representing images
        labels: numpy array representing labels
        test_ratio: float
            ratio of dataset used for test

    Returns:
        a tuple of image and label tuple
        (train_imgs, train_labels), (test_imgs, test_labels)
    """
    indices = list(range(len(imgs)))
    random.shuffle(indices)

    size = int(len(imgs) * test_ratio)
    train_idx = indices[size:]
    test_idx = indices[:size]

    train = imgs[train_idx], labels[train_idx]
    test = imgs[test_idx], labels[test_idx]

    return train, test


def convert_image_to_tfrecord(dirpath, imgs, labels,
                              filename='file_{}.tfrecord',
                              data_size=-1,
                              augmentation_ratio=0.0,
                              flip=False,
                              central_crop=None,
                              saturation=None,
                              test_ratio=0.2):
    """Convert tensor to TFRecord file.

    Args:
        dirpath: str
            target directory to save data
        imgs: numpy array or tensor
            target train/test data
        labels: numpy array or tensor
            label data
        filename: str
            file name to save as tfrecord
            default is `file_${number}.tfrecord`
            if filename contains `{}`,
            the ids are put there,
            otherwise, it goes right before extention.
            For instance, `filename = 'sample_{}_data.tfrecord'`,
            the first file name will be `sample_1_data.tfrecord`.
            if extention `.tfrecord` is not used or lacked,
            add it to the file name.
        data_size: int
            number of data to split input data
            default is -1 (all are put in one file)
        augmentation_ratio: float
            propotion of dataset by this ratio is used for
            image augmentation on training dataset
            if set to 0.0, no augmentation even if other params
            such as `flip` is set to True.
        flip: bool
            set to True and it generates fliped image in training dataset
        central_crop: float
            ratio of cropping, range is (0, 1] (use `tf.image.central_crop`)
        saturation: float
            factor to multiply the saturation by
            (use `tf.image.adjust_saturation`)
        test_ratio: float (default: 0.2)
            ratio of dataset saved as test set

    Returns:
        None

    Raises:
        ValueError: raise when imgs and labels size do not match
        ValueError: raise if `test` is not in `[0.0, 1.0]`
    """
    if imgs.shape[0] != labels.shape[0]:
        raise ValueError(f'Data size does not match: {imgs.shape} and {labels.shape}')

    if not filename.endswith('.tfrecord'):
        filename += '.tfrecord'

    if '{}' not in filename:
        name, ext = os.path.splitext(filename)
        filename = name + '_{}' + ext

    if test_ratio > 0.:
        train, test = train_test_split(imgs, labels, test_ratio)
    else:
        train = (imgs, labels)


    def _count_files(imgs):
        nonlocal data_size

        if data_size > 0:
            num_files = math.ceil(imgs.shape[0] / data_size)
            batch_size = data_size
        else:
            num_files = 1
            batch_size = imgs.shape[0]
        return num_files, batch_size


    def _batch_writer_to_tfrecord(imgs, labels,
                                  num_files,
                                  data_size,
                                  dirpath,
                                  filename,
                                  type):

        # write train data to tfrecord
        start = 0
        if not os.path.isdir(os.path.join(dirpath, 'train')):  # pragma: no cover
            os.makedirs(os.path.join(dirpath, 'train'))

        for i in range(num_files):
            # grouping by batch
            img_batch = imgs[start:start+data_size]
            label_batch = labels[start:start+data_size]
            start += data_size

            fname = filename.format(f'{i:02d}')
            write_to_tfrecord(img_batch, label_batch, os.path.join(dirpath, type, fname))


    imgs, labels = train
    num_files, batch_size = _count_files(imgs)

    _batch_writer_to_tfrecord(imgs, labels, num_files, batch_size, dirpath, filename, 'train')

    # write generated data separately to tfrecord
    if augmentation_ratio > 0:  # pragma: no cover
        new_imgs, new_labels = _data_augmentation(
            imgs, labels,
            ratio=augmentation_ratio,
            flip=flip,
            central_crop=central_crop,
            saturation=saturation,
        )
        train = new_imgs, new_labels

        # no longer needed
        del imgs, labels

        num_files, batch_size = _count_files(new_imgs)
        l, r = filename.split('{}')
        aug_filename = l + 'aug_{}' + r

        _batch_writer_to_tfrecord(new_imgs, new_labels, num_files, data_size, dirpath, aug_filename, 'train')

        del new_imgs, new_labels, train

    # write test data to tfrecord
    if test_ratio > 0.:
        start = 0

        if not os.path.isdir(os.path.join(dirpath, 'test')):  # pragma: no cover
            os.makedirs(os.path.join(dirpath, 'test'))

        imgs, labels = test
        num_files, batch_size = _count_files(imgs)

        _batch_writer_to_tfrecord(imgs, labels, num_files, batch_size, dirpath, filename, 'test')


def load_tfrecord_files(dirpath, filename='*.tfrecord'):  # pragma: no cover
    files = glob.glob(os.path.join(dirpath, filename))
    dataset = read_from_tfrecord(files)
    return dataset


def data_generator(filepaths, buffer_size=6000, shuffle=True, repeat=False, batch_size=32):
    """Generate data batch.

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
        imgs = tf.io.decode_raw(data['image'], out_type=tf.uint8)
        imgs = tf.cast(tf.reshape(imgs, shape=(-1, 224, 224, 3)), dtype=tf.float32)
        imgs = tf.keras.applications.mobilenet.preprocess_input(imgs)
        labels = tf.io.decode_raw(data['label'], out_type=tf.uint8)
        yield imgs, tf.squeeze(labels)
