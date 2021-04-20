#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from tests.ml.test_infer import DogBreedClassificationInferenceModelTest
from tests.ml.test_labels import DogBreedClassificationLabelDataTest
from tests.services.test_handlers import HandlersTest


test_cases = (
    DogBreedClassificationLabelDataTest,
    DogBreedClassificationInferenceModelTest,
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
