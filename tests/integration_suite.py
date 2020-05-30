#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from tests.utils import build_testsuite
from tests.models.test_mixins import BasicModelFeatureTest
from tests.models.test_image import ImageModelTest
from tests.ml.test_data import DataGeneratorIntegrationTest


def suite():
    test_cases = (
        BasicModelFeatureTest,
        ImageModelFeatureTest,
        DataGeneratorIntegrationTest,
    )

    suite = build_testsuite(test_cases)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
