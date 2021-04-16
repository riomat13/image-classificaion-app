#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, Mock

import io
import json
import logging
import os
import tempfile

import numpy as np
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError

from image_app.app import create_app
from image_app.orm.db import drop_db, reset_db, setup_session
from image_app.settings import ROOT_DIR, get_config

from tests.services.test_handlers import FakeInferenceModel, FakeLabelData

logging.disable(logging.CRITICAL)

label_list = ['dog', 'cat', 'lion', 'tiger', 'jaguar', 'giraffe', 'panda', 'fox', 'bear', 'bird']


class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app('test')
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        setup_session(get_config())

    def setUp(self):
        reset_db()
        self.client = self.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()
        drop_db()


class UploadImageTest(_Base):

    @patch('image_app.web.api.DogBreedClassificationLabelData.get_label_data')
    def test_send_label_name(self, get_label_data):
        mock_label_getter = Mock()
        mock_label_getter.get_label_by_id.side_effect = lambda x: label_list[x]

        get_label_data.return_value = mock_label_getter

        res = self.client.get(
            '/api/v1/prediction/label/3'
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json.get('label'), label_list[3])

    @patch('image_app.web.api.DogBreedClassificationLabelData.get_label_data')
    def test_send_label_list(self, get_label_data):
        get_label_data.return_value.id2label = label_list

        res = self.client.get(
            '/api/v1/prediction/labels/list'
        )

        res_list = res.json.get('labelList')

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res_list, label_list)

    def test_image_uploaded(self):
        res = self.client.get(
            '/api/v1/prediction/upload/image'
        )
        # should not be found if invalid request type
        self.assertEqual(res.status_code, 404)

        image = Image.new('RGB', size=(100, 100), color=(0, 0, 0))

        with tempfile.NamedTemporaryFile(delete=False) as test_file:
            image.save(test_file.name, 'JPEG')
            image.close()

            res = self.client.post(
                '/api/v1/prediction/upload/image',
                data=dict(file=(test_file, test_file.name)),
                content_type='multipart/form-data'
            )

        self.assertEqual(res.status_code, 201)
        self.assertIsNotNone(res.json.get('imgId'))

    def test_handle_failed_saving_image(self):
        test_data = io.BytesIO(b'test')
        res = self.client.post(
            '/api/v1/prediction/upload/image',
            data=dict(file=(test_data, 'init.jpg')),
            content_type='multipart/form-data'
        )

        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json.get('message'), 'failed to save image')

    def test_handle_different_content_type(self):
        # invalid content type
        res = self.client.post(
            '/api/v1/prediction/upload/image',
            data={'data': '', 'file': ''},
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 400)
        self.assertEqual('400 bad request', res.json.get('error'))
        self.assertIn('message', res.json)


class ServingModelAPITest(_Base):

    @patch('image_app.web.api.DogBreedClassificationLabelData', FakeLabelData)
    @patch('image_app.web.api.DogBreedClassificationInferenceModel', FakeInferenceModel)
    @patch('image_app.web.api.upload_image_file_object', return_value=({}, 201))
    @patch('image_app.web.api.Image')
    def test_serve_result(self, MockImage, *args):
        model = Mock()
        model.filename = 'sample.jpg'
        MockImage.get_by_encoded_id.return_value = model

        test_data = io.BytesIO(b'')

        res = self.client.post(
            '/api/v1/prediction/serve',
            data=dict(file=(test_data, 'init.jpg')),
            content_type='multipart/form-data'
        )

        data = res.json

        self.assertEqual(res.status_code, 201)
        self.assertEqual(len(data.get('result')), 3)
        self.assertEqual(len(data.get('result')[0]), 2)

    def test_handle_error_by_image_not_found(self):
        # when file could not be found in a request
        res = self.client.post(
            '/api/v1/prediction/serve',
            data=dict(file=(None, 'init.jpg')),
            content_type='multipart/form-data'
        )

        data = res.json
        self.assertEqual(res.status_code, 400)
        self.assertEqual(data.get('status'), 'error')

    @patch('image_app.web.api.DogBreedClassificationInferenceModel.infer', side_effect=RuntimeError)
    @patch('image_app.web.api.upload_image_file_object', return_value=({}, 201))
    def test_handle_error_during_inference(self, _, infer_image):
        test_data = io.BytesIO(b'test')

        res = self.client.post(
            '/api/v1/prediction/serve',
            data=dict(file=(test_data, 'init.jpg')),
            content_type='multipart/form-data'
        )

        self.assertEqual(res.status_code, 500)
        self.assertEqual(res.json.get('status'), 'error')


if __name__ == '__main__':
    unittest.main()
