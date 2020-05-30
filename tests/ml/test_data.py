#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

import math
import io
import functools
import random

import numpy as np
import tensorflow as tf
from PIL import Image

from image_app.ml.data import (
    get_label_from_path,
    load_images_all,
    load_images_all_by_batch,
    convert_images_to_tensor,
    train_test_split,
    data_generator
)
from image_app.ml.serializer import tf_serialize_example


logging.disable(logging.CRITICAL)


def generate_random_images(image_shape=(224, 224, 3), n=10):
    image_size = functools.reduce(lambda x, y: x * y, image_shape)
    imgs = np.random.randint(0, 255, n * image_size, dtype=np.uint8) \
            .reshape(-1, *image_shape)
    return imgs


def generate_img_file_from_arr(arr):
    im = Image.fromarray(arr)
    f = io.BytesIO()
    im.save(f, format="PNG")
    return f


def generate_img_file(image_shape=(224, 224, 3)):
    image_size = functools.reduce(lambda x, y: x * y, image_shape)
    img = np.random.randint(0, 255, image_size, dtype=np.uint8) \
            .reshape(*image_shape)

    return generate_img_file_from_arr(img)


def generate_img_files(image_shape=(224, 224, 3), n=10):
    """Generate sample image file stored as BytesIO.

    Generate `n` files and return them as a list.
    """
    files = [generate_img_file(image_shape) for _ in range(n)]
    return files


class LoadImageFilesTest(unittest.TestCase):

    @patch('image_app.ml.base.LabelData.get_label_data')
    def test_label_getter(self, get_label):
        label_data = MagicMock()
        label_data.__getitem__.side_effect = lambda x: x
        get_label.return_value = label_data

        raw_labels = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american staffordshire_terrier', 'appenzeller', 'Australian_Terrier', 'BASENJI', 'bASSET', 'Beagle']

        labels = ['Affenpinscher', 'Afghan hound', 'African hunting dog', 'Airedale', 'American staffordshire terrier', 'Appenzeller', 'Australian terrier', 'Basenji', 'Basset', 'Beagle']

        path = 'data/some_dir/{}/test_img.jpg'

        for label, target in zip(raw_labels, labels):
            l = get_label_from_path(path.format(label))
            self.assertEqual(l, target)

    def test_convert_tensor_from_image_files(self):
        img_dataset = [Image.fromarray(arr) for arr in generate_random_images()]
        target_size = len(img_dataset)

        tensor = convert_images_to_tensor(img_dataset)

        # check generated tensor's shape, type, and if normalized
        self.assertEqual(tensor.shape, (target_size, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(abs(np.mean(tensor)) < 1.)

    @patch('image_app.ml.data.get_label_from_path', return_value=1)
    @patch('image_app.ml.data.load_image')
    @patch('image_app.ml.data.random.shuffle')
    def test_load_multiple_images(self, shuffle, load_image, get_label):
        imgs = generate_random_images(n=10)
        dummies = [None] * 10
        img_dataset = iter([Image.fromarray(arr) for arr in imgs])
        load_image.side_effect = lambda x, *args, **kwargs: next(img_dataset)

        imgs, labels = load_images_all(dummies, shuffle=True)

        self.assertEqual(imgs.shape[0], labels.shape[0])

        self.assertEqual(imgs.shape, (10, 224, 224, 3))

        shuffle.assert_called_once_with(dummies)

    @patch('image_app.ml.data.load_image')
    @patch('image_app.ml.data.get_label_from_path', return_value=1)
    def test_handle_error_during_read_image(self, get_label, load_image):
        img_shape = (224, 224, 3)
        img_size = 224 * 224 * 3
        files = generate_img_files(img_shape, n=10)

        failed = 0

        def mock_load_img(*args, **kwargs):
            nonlocal failed

            prob = random.random()
            if prob > 0.5:
                return np.random.randint(0, 255, img_size) \
                        .astype(np.float32) \
                        .reshape(*img_shape)

            else:
                failed += 1
                raise ValueError('random fail')

        load_image.side_effect = mock_load_img

        imgs, labels = load_images_all([None] * 10)
        self.assertEqual(imgs.shape[0], labels.shape[0])
        reduce_summed = np.sum(imgs, axis=(1, 2, 3))
        if 0 in reduce_summed:
            self.fail('Empty image should not be contained')

        target_size = len(files) - failed
        self.assertEqual(imgs.shape, (target_size, 224, 224, 3))

    @patch('image_app.ml.data.get_label_from_path', return_value=1)
    @patch('image_app.ml.data.load_image')
    @patch('image_app.ml.data.random.shuffle')
    def test_load_multiple_images_by_batch(self, shuffle, load_image, get_label):
        with self.assertRaises(ValueError):
            next(load_images_all_by_batch([None] * 100, batch_size=0))

        imgs = generate_random_images(n=10)
        dummies = [None] * 10
        img_dataset = (Image.fromarray(arr) for arr in imgs)
        load_image.side_effect = lambda x, *args, **kwargs: next(img_dataset)

        batch_size = 4
        gen = load_images_all_by_batch(dummies, shuffle=True, batch_size=batch_size)
        self.assertTrue(hasattr(gen, '__next__'))

        rest = 10

        for i, (batch_imgs, batch_labels) in enumerate(gen):
            with self.subTest(i=i):
                self.assertEqual(batch_imgs.shape[0], batch_labels.shape[0])
                self.assertEqual(batch_imgs.shape, (min(batch_size, rest), 224, 224, 3))
            rest -= batch_size

        # check when batch size is larger than actual dataset size
        img_dataset = (Image.fromarray(arr) for arr in imgs)
        gen = load_images_all_by_batch(dummies, shuffle=False, batch_size=1000)
        imgs, labels = next(gen)
        self.assertEqual(imgs.shape[0], labels.shape[0])
        self.assertEqual(imgs.shape, (10, 224, 224, 3))

        with self.assertRaises(StopIteration):
            next(gen)

        # shuffle called once when set to True
        shuffle.assert_called_once_with(dummies)


class DataSpliterTest(unittest.TestCase):

    def test_split_files_into_train_and_test(self):
        filename = 'sample_{}.jpg'
        data = [filename.format(i) for i in range(100)]
        train, test = train_test_split(data, 0.2)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

        self.assertEqual(len(set(train + test)), 100)


class DataGeneratorIntegrationTest(unittest.TestCase):

    def test_data_generator(self):
        imgs = np.random.randint(0, 255, 100*224*224*3, dtype=np.uint8).reshape(100, 224, 224, 3)
        labels = np.random.randint(0, 8, 100, dtype=np.uint8)

        filepath = '/tmp/tmp_data_generator_test.tfrecord'

        # reduce time to rewrite each test
        if not os.path.isfile(filepath):
            dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
            serialized_dataset = dataset.map(tf_serialize_example)

            writer = tf.data.experimental.TFRecordWriter(filepath)
            writer.write(serialized_dataset)

        count = 0
        for batch_imgs, batch_labels in data_generator([filepath], batch_size=10):
            count += 1
            self.assertEqual(batch_imgs.shape, (10, 224, 224, 3))
            self.assertEqual(batch_imgs.dtype, tf.float32)

            batch_imgs = np.sum(batch_imgs, axis=(1, 2, 3))
            self.assertNotIn(0, batch_imgs)
            self.assertEqual(batch_labels.shape, (10,))
            self.assertEqual(batch_labels.dtype, tf.uint8)


if __name__ == '__main__':
    unittest.main()
