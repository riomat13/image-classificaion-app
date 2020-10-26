#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dotenv import load_dotenv

from ._base import Config, ROOT_DIR

load_dotenv(ROOT_DIR / '.env')


class _DB_URI_PROPERTY(object):
    # construct DB URI
    def __init__(self):
        self.uri = None

    def __get__(self, instance, owner):
        if self.uri is None:
            user = owner.DATABASE_USERNAME
            pwd = owner.DATABASE_PASSWORD
            host = owner.DATABASE_HOST
            port = owner.DATABASE_PORT
            db = owner.DATABASE_NAME

            self.uri =  f'postgres://{user}:{pwd}@{host}:{port}/{db}'
        return self.uri


class ProductionConfig(Config):
    ENV = 'production'

    PROPAGATE_EXCEPTIONS = False

    DATABASE_URI = _DB_URI_PROPERTY()
