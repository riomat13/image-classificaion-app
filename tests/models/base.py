#!/usr/bin/env python3

import unittest

from image_app.app import create_app
from image_app.orm.db import get_engine, setup_session, reset_db, drop_db


class BaseTestCase(unittest.TestCase):

    def setUp(self):
        app = create_app('test')
        self.app_context = app.app_context()
        self.app_context.push()

        reset_db()

    def tearDown(self):
        self.app_context.pop()
        drop_db()
