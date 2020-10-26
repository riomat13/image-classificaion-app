#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import io
import tempfile

from image_app.app import create_app
from image_app.models._utils import generate_code
from image_app.orm.db import reset_db, drop_db
from image_app.models.image import Image


class ImageModelFeatureTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app = create_app('test')
        cls.app_context = app.app_context()
        cls.app_context.push()

        reset_db()

        cls.img = Image(filename='test.jpg')
        cls.img.save()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()
        drop_db()

    def test_validate_encoded_string_by_id(self):
        encoded = self.img.get_encode()
        self.assertTrue(isinstance(encoded, str))
        self.assertTrue(self.img.validate_code(encoded))

    def test_set_processed_mark(self):
        self.assertFalse(self.img.processed)
        self.assertIsNone(self.img.processed_at)

        self.img.set_processed()
        self.assertTrue(self.img.processed)
        self.assertIsNotNone(self.img.processed_at)

    def test_fetch_model_by_encoded_id(self):
        encoded_id = generate_code(self.img.id)
        model = Image.get_by_encoded_id(encoded_id)
        self.assertIsNotNone(model)
        self.assertEqual(model.id, self.img.id)


if __name__ == '__main__':
    unittest.main()
