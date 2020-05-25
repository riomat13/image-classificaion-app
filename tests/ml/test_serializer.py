#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This is a simple test to make sure preprocess pipeline works
# This will not be included in unittest nor integration test suites,
# since this is not for long-running.

import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from image_app.ml.serializer import read_from_tfrecord, write_to_tfrecord


imgs = np.random.randint(0, 255, 100 * 30 * 30 * 3, dtype=np.uint8) \
        .reshape(-1, 30, 30, 3)
labels = np.random.randint(0, 8, 100, dtype=np.uint8)

filepath = '/tmp/tmp.tfrecord'


class WriteReadTFRecordTest(unittest.TestCase):

    def test_write_and_read_tfrecord(self):
        write_to_tfrecord(imgs, labels, filepath)

        self.assertTrue(os.path.isfile(filepath))

        dataset = read_from_tfrecord([filepath])
        chunks = []

        for d in dataset:
            chunks.append(d)

        self.assertEqual(len(chunks), 100)

        img = tf.io.decode_raw(chunks[0]['image'], out_type=tf.uint8)
        img = tf.reshape(img, shape=(30, 30, 3))
        label = tf.io.decode_raw(chunks[0]['label'], out_type=tf.uint8)

        self.assertEqual(img.shape, (30, 30, 3))
        self.assertEqual(label.shape, (1,))

        chunks = []
        for d in dataset.batch(10):
            chunks.append(d)

        self.assertEqual(len(chunks), 10)

        img = tf.io.decode_raw(chunks[0]['image'], out_type=tf.uint8)
        img = tf.reshape(img, shape=(-1, 30, 30, 3))
        label = tf.io.decode_raw(chunks[0]['label'], out_type=tf.uint8)

        self.assertEqual(img.shape, (10, 30, 30, 3))
        self.assertEqual(label.shape, (10, 1))


if __name__ == '__main__':
    unittest.main()
