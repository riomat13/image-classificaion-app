#!/usr/bin/env python3

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
    LabelData,
    get_label_from_path,
    load_images_all,
    convert_images_to_tensor,
    train_test_split,
    convert_image_to_tfrecord,
    load_tfrecord_files,
    data_generator
)
from image_app.ml.serializer import write_to_tfrecord


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


class LabelDataInstanceTest(unittest.TestCase):

    def test_object_creaton_by_instantiation(self):
        labels = LabelData.get_label_data()

        with self.assertRaises(RuntimeError):
            LabelData()

        labels2 = LabelData.get_label_data()
        self.assertTrue(labels is labels2)

    def test_label_set_creation(self):
        labels = LabelData.get_label_data()
        self.assertEqual(len(labels), 121)

        # match order in iteration and
        # fetching by index or by label name
        for i, label in enumerate(labels):
            self.assertEqual(label, labels[i])
            self.assertEqual(i, labels[label])

    def test_label_sets(self):
        labels = LabelData.get_label_data()

        self.assertEqual(labels.get_label_count(), 121)

        # handle nested label
        target = [0, 1, 2, 3, [4, 5], 'Affenpinscher']
        label_names = labels[target]
        self.assertEqual(len(label_names), len(target))
        self.assertEqual(len(label_names[4]), 2)
        self.assertEqual(label_names[0], 'others')
        self.assertEqual(label_names[-1], 1)

        with self.assertRaises(TypeError):
            labels[10.0]

        with self.assertRaises(TypeError):
            labels[(10, 20, 10.5, 'Affenpinscher')]

class LoadingImageFilesTest(unittest.TestCase):

    def test_label_getter(self):
        labels = ['Affenpinscher', 'Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_staffordshire_terrier', 'Appenzeller', 'Australian_terrier', 'Basenji', 'Basset', 'Beagle']

        path = 'data/some_dir/{}/test_img.jpg'

        for i, label in enumerate(labels, 1):
            l = get_label_from_path(path.format(label))
            self.assertEqual(l, i)

        labels = ['airplane', 'dog', 'music']
        for i, label in enumerate(labels):
            l = get_label_from_path(path.format(label))
            self.assertEqual(l, 0)

    def test_convert_tensor_from_image_files(self):
        img_dataset = [Image.fromarray(arr) for arr in generate_random_images()]
        target_size = len(img_dataset)

        tensor = convert_images_to_tensor(img_dataset)

        # check generated tensor's shape, type, and if normalized
        self.assertEqual(tensor.shape, (target_size, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(abs(np.mean(tensor)) < 1.)


    @patch('image_app.ml.data.glob.glob')
    @patch('image_app.ml.data.get_label_from_path', return_value=1)
    @patch('image_app.ml.data.tf.keras.preprocessing.image.load_img')
    def test_load_multiple_images(self, load_img, get_label, glob):
        imgs = generate_random_images()
        glob.return_value = [None] * len(imgs)
        img_dataset = iter([Image.fromarray(arr) for arr in imgs])
        load_img.side_effect = lambda x, *args, **kwargs: next(img_dataset)

        imgs, labels = load_images_all('/tmp')

        self.assertEqual(imgs.shape[0], labels.shape[0])

        self.assertEqual(imgs.shape, (10, 224, 224, 3))

        # should raise if negative depth
        with self.assertRaises(ValueError):
            load_images_all('/tmp', depth=-1)

    @patch('image_app.ml.data.glob.glob')
    @patch('image_app.ml.data.tf.keras.preprocessing.image.load_img')
    @patch('image_app.ml.data.get_label_from_path', return_value=1)
    def test_handle_error_during_read_image(self, get_label, load_img, glob):
        img_shape = (224, 224, 3)
        img_size = 224 * 224 * 3
        files = generate_img_files(img_shape)
        glob.return_value = files

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

        load_img.side_effect = mock_load_img

        imgs, labels = load_images_all('/tmp')
        self.assertEqual(imgs.shape[0], labels.shape[0])

        target_size = len(files) - failed
        self.assertEqual(imgs.shape, (target_size, 224, 224, 3))


class DataSpliterTest(unittest.TestCase):

    def test_split_with_randomness(self):
        imgs = generate_random_images()
        labels = np.random.randint(0, 8, imgs.shape[0])

        test_ratio = 0.2
        target_test = int(len(imgs) * test_ratio)
        target_train = len(imgs) - target_test
        target_sizes = [target_train, target_test]

        train1, test1 = train_test_split(imgs, labels, test_ratio)

        self.assertEqual(len(train1[0]) + len(test1[0]), len(imgs))
        self.assertEqual(len(train1[1]) + len(test1[1]), len(imgs))

        train2, test2 = train_test_split(imgs, labels, test_ratio)

        self.assertEqual(len(train2[0]) + len(test2[0]), len(imgs))
        self.assertEqual(len(train2[1]) + len(test2[1]), len(imgs))

        self.assertEqual(train1[0].shape, train2[0].shape)
        self.assertEqual(test1[0].shape, test2[0].shape)
        self.assertFalse(np.array_equal(train1[0], train2[0]))
        self.assertFalse(np.array_equal(test1[0], test2[0]))


class ConvertToTFRecordTest(unittest.TestCase):

    @patch('image_app.ml.data.write_to_tfrecord')
    def test_save_to_tfrecord_file(self, writer):
        # without saving test data
        imgs = generate_random_images()
        labels = np.random.randint(0, 8, imgs.shape[0])

        convert_image_to_tfrecord('/tmp', imgs, labels, filename='test', test_ratio=0.0)

        writer.assert_called_once()
        args = writer.call_args_list[0][0]

        self.assertTrue(np.array_equal(imgs, args[0]))
        self.assertTrue(np.array_equal(labels, args[1]))
        self.assertEqual(args[2], '/tmp/train/test_00.tfrecord')

    @patch('image_app.ml.data.write_to_tfrecord')
    @patch('image_app.ml.data.train_test_split')
    def test_save_train_test_dataset(self, train_test_split, writer):
        imgs = generate_random_images(n=20)
        labels = np.random.randint(0, 8, imgs.shape[0]).reshape(-1, 1)

        test_ratio = 0.2
        target_test = int(len(imgs) * test_ratio)
        target_train = len(imgs) - target_test
        target_sizes = [target_train, target_test]

        train_test_split.return_value = (imgs[:target_train], labels[:target_train]), (imgs[target_train:], labels[target_train:])

        convert_image_to_tfrecord('/tmp', imgs, labels, filename='test', test_ratio=test_ratio)

        self.assertEqual(writer.call_count, 2)

        # for train
        args = writer.call_args_list[0][0]

        # 80% of total data
        self.assertEqual(args[0].shape, (target_sizes[0], 224, 224, 3))
        self.assertEqual(args[1].shape, (target_sizes[0], 1))
        self.assertEqual(args[2], '/tmp/train/test_00.tfrecord')

        # for test
        args = writer.call_args_list[1][0]

        self.assertEqual(args[0].shape, (target_sizes[1], 224, 224, 3))
        self.assertEqual(args[1].shape, (target_sizes[1], 1))
        self.assertEqual(args[2], '/tmp/test/test_00.tfrecord')


    def test_fail_when_pass_imbalanced_data_to_convert_image_to_tfrecord(self):
        imgs = generate_random_images()
        labels = np.random.randint(1, 8, imgs.shape[0] - 2)

        with self.assertRaises(ValueError):
            convert_image_to_tfrecord('/tmp', imgs, labels)

        labels = np.random.randint(1, 8, imgs.shape[0] + 3)

        with self.assertRaises(ValueError):
            convert_image_to_tfrecord('/tmp', imgs, labels)

    @patch('image_app.ml.data.tf.data')
    @patch('image_app.ml.data.write_to_tfrecord')
    def test_save_to_multiple_tfrecord_files(self, writer, tfdata):
        data_size = 6
        n_imgs = 20
        expected_called = math.ceil(n_imgs / data_size)

        imgs = generate_random_images(n=n_imgs)
        labels = np.random.randint(1, 8, imgs.shape[0])

        # test for train dataset
        convert_image_to_tfrecord('/tmp', imgs, labels,
                                  filename='ff.tfrecord',
                                  data_size=data_size,
                                  test_ratio=0.0)

        # test with one batch so that entire data should be passed
        self.assertEqual(writer.call_count, expected_called)

        # count number of images in each batch
        total_imgs = 0
        for args, _ in writer.call_args_list:
            total_imgs += args[0].shape[0]

        self.assertEqual(total_imgs, n_imgs)


class DataGeneratorIntegrationTest(unittest.TestCase):

    def test_data_generator(self):
        imgs = np.random.randint(0, 255, 100 * 224 * 224 * 3, dtype=np.uint8) \
                .reshape(-1, 224, 224, 3)
        labels = np.random.randint(0, 8, 100, dtype=np.uint8)
        filepath = '/tmp/tmp.tfrecord'

        write_to_tfrecord(imgs, labels, filepath)

        count = 0
        for imgs_, labels_ in data_generator([filepath], batch_size=10):
            count += 1
            self.assertEqual(imgs_.shape, (10, 224, 224, 3))
            self.assertEqual(imgs_.dtype, tf.float32)
            self.assertEqual(labels_.shape, (10,))
            self.assertEqual(labels_.dtype, tf.uint8)


if __name__ == '__main__':
    unittest.main()
