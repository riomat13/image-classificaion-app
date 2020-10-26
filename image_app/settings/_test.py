#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dotenv import load_dotenv

from ._base import Config, ROOT_DIR

load_dotenv(ROOT_DIR / '.env.dev')


class TestConfig(Config):
    DEBUG = True
    TESTING = True

    ENV = 'test'

    UPLOAD_DIR = 'media/test'

    # TODO: can use in memory DB?
    DATABASE_URI = 'sqlite:////tmp/test.db'
