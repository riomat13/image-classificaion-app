#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from tests.integration_suite import test_cases as itest_cases
from tests.unittest_suite import test_cases as utest_cases


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_cls in utest_cases + itest_cases:
        tests = loader.loadTestsFromTestCase(test_cls)
        suite.addTests(tests)

    return suite


if __name__ == '__main__':
    unittest.main()
