#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path


ROOT_DIR = Path(__file__).parents[2]


class EnvironmentProperty(object):
    """Property to load lazily."""
    def __init__(self, key, default=None):
        self.key = key
        self.value = None
        self.default = default

    def __get__(self, instance, owner):
        if self.value is None:
            self.value = os.environ.get(self.key, self.default)
        return self.value


class MLMODEL_PROPERTY(object):

    def __init__(self):
        self.map = None

    def __get__(self, instance, owner):
        if self.map is None:
            self.map = {
                'DOG_BREED': {
                    'MODEL_DATA': os.environ.get('APP_MODEL_PATH', 'saved_model/base.tflite'),
                    'LABEL_DATA': os.path.join(ROOT_DIR, 'data/category_list.txt')
                },
            }
        return self.map


class Config(object):
    DEBUG = False
    TESTING = False

    SECRET_KEY = EnvironmentProperty('SECRET_KEY')

    # compiled frontend directory
    STATIC_DIR = os.path.join(ROOT_DIR, 'frontend/dist')
    UPLOAD_DIR = 'media/uploaded'

    # API
    JSONIFY_MIMETYPE = 'application/json'

    # ML Model data file paths
    ML_MODELS = MLMODEL_PROPERTY()

    # database settings
    DATABASE_NAME = EnvironmentProperty('FLASK_DB_NAME')
    DATABASE_HOST = EnvironmentProperty('FLASK_DB_HOST')
    DATABASE_PORT = EnvironmentProperty('FLASK_DB_PORT')
    DATABASE_USERNAME = EnvironmentProperty('FLASK_DB_USER')
    DATABASE_PASSWORD = EnvironmentProperty('FLASK_DB_PASSWORD')
    DATABASE_URI = EnvironmentProperty('FLASK_DB_URI')

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return f'{self.__file__}.{self.__name__}'

    @staticmethod
    def init_app(app):
        pass
