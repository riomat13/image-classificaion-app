#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ._base import Config


class TestConfig(Config):
    DEBUG = True
    TESTING = True

    ENV = 'test'

    UPLOAD_DIR = 'media/test'

    DATABASE_URI = 'sqlite:///app-test.db'
