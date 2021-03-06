#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, Mock

import logging
import os
import io
import json

import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from image_app.app import create_app
from image_app.settings import ROOT_DIR

logging.disable(logging.CRITICAL)

label_list = ['dog', 'cat', 'lion', 'tiger', 'jaguar', 'giraffe', 'panda', 'fox', 'bear', 'bird']


class _Base(unittest.TestCase):

    @classmethod

    def setUp(self):
        self.app = create_app('test')
        self.app_context = self.app.app_context()
        self.app_context.push()

        self.client = self.app.test_client()

    def tearDown(self):
        self.app_context.pop()


class UploadImageTest(_Base):

    @patch('image_app.web.api.LabelData.get_label_data')
    def test_send_label_name(self, get_label_data):
        mock_label_getter = Mock()
        mock_label_getter.get_label_by_id.side_effect = lambda x: label_list[x]

        get_label_data.return_value = mock_label_getter

        res = self.client.get(
            '/api/v1/prediction/label/3'
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json.get('label'), label_list[3])


    @patch('image_app.web.api.LabelData.get_label_data')
    def test_send_label_list(self, get_label_data):
        get_label_data.return_value.id2label = label_list

        res = self.client.get(
            '/api/v1/prediction/labels/list'
        )

        res_list = res.json.get('labelList')

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res_list, label_list)


    @patch('image_app.web.api.os.path.join', return_value='/tmp/test.jpg')
    @patch('image_app.web.api.secure_filename', return_value='test.jpg')
    @patch('image_app.web.api.Image')
    def test_image_uploaded(self, Image, secure_filename, *args):
        res = self.client.get(
            '/api/v1/prediction/upload/image'
        )
        # should not be found if invalid request type
        self.assertEqual(res.status_code, 404)

        mock_model = Mock()
        Image.return_value = mock_model
        target_id = 10
        Image.query.count.return_value = target_id
        mock_model.get_encode.return_value = 'encoded'

        test_data = io.BytesIO(b'test')
        test_data.save = Mock()

        res = self.client.post(
            '/api/v1/prediction/upload/image',
            data=dict(file=(test_data, 'init.jpg')),
            content_type='multipart/form-data'
        )

        self.assertEqual(res.status_code, 201)
        self.assertIsNotNone(res.json.get('imgId'))
        secure_filename.assert_called_once_with('init.jpg')
        mock_model.get_encode.assert_called_once()

    @patch('image_app.web.api.Image')
    def test_handle_failed_saving_image(self, Image):
        mock_model = Mock()
        mock_model.save.side_effect = SQLAlchemyError
        Image.return_value = mock_model
        Image.query.count.return_value = 1

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

    @patch('image_app.web.api.infer_image')
    @patch('image_app.web.api.np.frombuffer')
    @patch('image_app.web.api.load_image')
    @patch('image_app.web.api.secure_filename')
    @patch('image_app.web.api.os.path.splitext')
    def test_serve_result(self, mock_splitext, _a, load_image, _b, infer_image):
        infer_image.return_value = [
            ('sample1', 0.12345),
            ('sample2', 0.12345),
            ('sample3', 0.12345),
        ]

        test_data = io.BytesIO(b'test')
        mock_splitext.return_value = 'init', '.jpg'
        load_image.return_value = \
            np.random.randint(0, 255, 224 * 224 * 3) \
                .reshape(224, 224, 3) \

        res = self.client.post(
            '/api/v1/prediction/serve',
            data=dict(file=(test_data, 'init.jpg')),
            content_type='multipart/form-data'
        )

        data = res.json

        self.assertEqual(res.status_code, 201)
        self.assertEqual(len(data.get('result')), 3)
        self.assertEqual(len(data.get('result')[0]), 2)
        self.assertEqual(data.get('result')[0][0], 'sample1')
        # convert to percentage
        self.assertEqual(data.get('result')[0][1], '12.35')

    @patch('image_app.web.api.infer_image')
    @patch('image_app.web.api.load_image')
    @patch('image_app.web.api.Image')
    def test_handle_error_by_image_not_found(self, Image, *args):
        Image.query.count.return_value = 0

        # when file could not be found in a request
        res = self.client.post(
            '/api/v1/prediction/serve',
            data=dict(file=(None, 'init.jpg')),
            content_type='multipart/form-data'
        )

        data = res.json
        self.assertEqual(res.status_code, 400)
        self.assertEqual(data.get('status'), 'error')

        Image.return_value.save.assert_not_called()

    @patch('image_app.web.api.infer_image', side_effect=RuntimeError)
    @patch('image_app.web.api._read_and_save_image', return_value=({}, 201))
    def test_handle_error_during_inference(self, _, infer_image):
        test_data = io.BytesIO(b'test')

        res = self.client.post(
            '/api/v1/prediction/serve',
            data=dict(file=(test_data, 'init.jpg')),
            content_type='multipart/form-data'
        )

        self.assertEqual(res.status_code, 500)
        self.assertEqual(res.json.get('status'), 'error')

    @patch('image_app.web.api.infer_image')
    @patch('image_app.web.api.Image.get_by_encoded_id')
    @patch('image_app.web.api.os.path')
    def test_serve_api_by_image_id(self, _, get_by_encoded_id, infer_image):
        infer_image.return_value = [
            ('sample1', 0.12345),
            ('sample2', 0.12345),
            ('sample3', 0.12345),
        ]

        get_by_encoded_id.return_value.filename = 'test'

        res = self.client.get(
            '/api/v1/prediction/serve/image_id',
            data=json.dumps({'imgId': 'test'}),
            content_type='application/json'
        )

        data = res.json

        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(data.get('result')), 3)
        self.assertEqual(len(data.get('result')[0]), 2)
        self.assertEqual(data.get('result')[0][0], 'sample1')
        # convert to percentage
        self.assertEqual(data.get('result')[0][1], '12.35')


    @patch('image_app.web.api.infer_image', side_effect=RuntimeError)
    @patch('image_app.web.api.Image.get_by_encoded_id')
    @patch('image_app.web.api.os.path')
    def test_handle_error_in_serve_api_by_image_id(self, _, get_by_encoded_id, infer_image):
        res = self.client.get(
            '/api/v1/prediction/serve/image_id',
            data=json.dumps({'imgId': 'test'}),
            content_type='application/json'
        )

        self.assertEqual(res.status_code, 500)
        self.assertEqual(res.json.get('status'), 'error')


if __name__ == '__main__':
    unittest.main()
