#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path


ROOT_DIR = Path(__file__).parents[2]


class Config(object):
    DEBUG = False
    TESTING = False

    SECRET_KEY = os.environ.get('SECRET_KEY')

    # compiled frontend directory
    STATIC_DIR = os.path.join(ROOT_DIR, 'frontend/build/static')
    TEMPLATE_DIR = os.path.join(ROOT_DIR, 'frontend/build')
    UPLOAD_DIR = 'media/uploaded'

    # API
    JSONIFY_MIMETYPE = 'application/json'

    # ML Model settings
    LABEL_LIST_PATH = os.path.join(ROOT_DIR, 'data/category_list.txt')
    MODEL_PATH = os.environ.get('APP_MODEL_PATH', 'saved_model/base.tflite')

    # database settings
    DATABASE_NAME = os.environ.get('FLASK_DB_NAME')
    DATABASE_HOST = os.environ.get('FLASK_DB_HOST')
    DATABASE_PORT = os.environ.get('FLASK_DB_PORT')
    DATABASE_USERNAME = os.environ.get('FLASK_DB_USER')
    DATABASE_PASSWORD = os.environ.get('FLASK_DB_PASSWORD')
    DATABASE_URI = os.environ.get('FLASK_DB_URI')

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return f'{self.__file__}.{self.__name__}'

    @staticmethod
    def init_app(app):
        pass
