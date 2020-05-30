#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os.path

from image_app.settings import ROOT_DIR


def suite():
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test*', top_level_dir=ROOT_DIR)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
