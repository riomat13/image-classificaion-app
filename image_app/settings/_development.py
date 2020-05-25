#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ._base import Config


class DevelopmentConfig(Config):
    DEBUG = True

    ENV = 'development'

    UPLOAD_DIR = 'media/dev'
