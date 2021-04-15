#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dotenv import load_dotenv

from ._base import Config, EnvironmentProperty, ROOT_DIR

load_dotenv(ROOT_DIR / '.env.dev')


class DevelopmentConfig(Config):
    DEBUG = True

    ENV = 'development'

    UPLOAD_DIR = 'media/dev'
