#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from tests.utils import build_testsuite
from tests.ml.test_data import (
    LabelDataInstanceTest,
    LoadingImageFilesTest,
    DataSpliterTest,
    ConvertToTFRecordTest,
)
from tests.ml.test_serializer import WriteReadTFRecordTest


def suite():
    test_cases = (
        LabelDataInstanceTest,
        LoadingImageFilesTest,
        DataSpliterTest,
        ConvertToTFRecordTest,
        WriteReadTFRecordTest,
    )

    suite = build_testsuite(test_cases)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
