#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from tests.ml.test_data import (
    LoadImageFilesTest,
    DataSpliterTest,
)
from tests.ml.test_infer import DogBreedClassificationInferenceModelTest
from tests.ml.test_labels import DogBreedClassificationLabelDataTest
from tests.ml.test_serializer import WriteReadTFRecordTest
from tests.services.test_handlers import HandlersTest


test_cases = (
    DogBreedClassificationLabelDataTest,
    DogBreedClassificationInferenceModelTest,
    LoadImageFilesTest,
    DataSpliterTest,
    WriteReadTFRecordTest,
    HandlersTest,
)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_cls in test_cases:
        tests = loader.loadTestsFromTestCase(test_cls)
        suite.addTests(tests)
    return suite


if __name__ == '__main__':
    unittest.main()
