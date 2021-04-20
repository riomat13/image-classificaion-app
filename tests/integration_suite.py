#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from tests.models.test_mixins import BasicModelFeatureTest
from tests.models.test_image import ImageModelFeatureTest
from tests.web.test_api import ServingModelAPITest, UploadImageTest


test_cases = (
    BasicModelFeatureTest,
    ImageModelFeatureTest,
    ServingModelAPITest,
    UploadImageTest,
)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_cls in test_cases:
        tests = loader.loadTestsFromTestCase(test_cls)
        suite.addTests(tests)
    return suite


if __name__ == '__main__':
    unittest.main()
