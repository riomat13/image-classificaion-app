#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from sqlalchemy import Column, Integer, String

from image_app.app import create_app
from image_app.orm.db import ModelBase, reset_db, drop_db
from image_app.models.mixins import BaseModelMixin
from tests.models.base import BaseTestCase


class BasicModelFeatureTest(unittest.TestCase):

    class SampleModel(BaseModelMixin, ModelBase):
        __tablename__ = 'sample'

        id = Column(Integer, primary_key=True)
        name = Column(String, unique=True)

    @classmethod
    def setUpClass(cls):
        app = create_app('test')
        app_context = app.app_context()
        app_context.push()
        reset_db()

        #import pdb; pdb.set_trace()
        model = cls.SampleModel(name='test')
        model.save()
        cls.target_id = model.id

        cls.app = app
        cls.app_context = app_context

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()
        drop_db()

    def test_creation_and_fetch_by_id(self):
        model = self.SampleModel.get(self.target_id)
        self.assertIsNotNone(model)
        self.assertEqual(model.id, self.target_id)

    def test_query_model_item(self):
        model = self.SampleModel.query.first()
        self.assertIsNotNone(model)
        self.assertEqual(model.id, self.target_id)

        model2 = model.query.get(model.id)
        self.assertTrue(model is model2)
