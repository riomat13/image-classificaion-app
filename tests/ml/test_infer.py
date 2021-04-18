#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, Mock

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image

from image_app.exception import ConfigAlreadySetError
from image_app.settings import set_config
try:
    set_config('test')
except ConfigAlreadySetError:
    pass

from image_app.ml.infer import DogBreedClassificationInferenceModel
from image_app.ml.labels import DogBreedClassificationLabelData


class DogBreedClassificationInferenceModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DogBreedClassificationInferenceModel()
        cls.label_size = DogBreedClassificationLabelData \
                .get_label_data() \
                .get_label_count()

    def test_make_inference(self):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        res = self.model.infer(img)

        self.assertEqual(res.shape, (self.label_size,))


if __name__ == '__main__':
    unittest.main()
