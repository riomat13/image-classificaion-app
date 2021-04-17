#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from image_app.settings import set_config
set_config('test')

import os
import tempfile

import numpy as np

from image_app.exception import InvalidImageDataFormat, InvalidMessage
from image_app.services import commands
from image_app.services.messagebus import messagebus
from image_app.ml.base import LabelData
from image_app.ml.infer import InferenceModel
from image_app.settings import ROOT_DIR


class FakeLabelData(LabelData):

    labels = ['test', 'infer', 'sample', 'handler']

    def get_label_by_id(self, id):
        return self.labels[id]

    def get_id_by_label(self, label):
        return ''

    @staticmethod
    def get_label_data():
        return FakeLabelData()


class FakeInferenceModel(InferenceModel):

    def infer(self, *args):
        data = np.random.randint(1, 100, len(FakeLabelData.labels)).astype(np.float32)
        data /= data.sum()
        return data


class HandlersTest(unittest.TestCase):

    def test_run_prediction(self):
        pred = messagebus.handle(
            commands.MakePrediction(
                image_path=os.path.join(ROOT_DIR, 'tests', 'data', 'sample.jpg'),
                model=FakeInferenceModel(),
            )
        )
        self.assertTrue(isinstance(pred, np.ndarray))

    def test_raise_error_due_to_wrong_image_format_when_predicting(self):
        fp = tempfile.NamedTemporaryFile()
        fp.write(b'non-image data')

        with self.assertRaises(InvalidImageDataFormat):
            messagebus.handle(
                commands.MakePrediction(
                    image_path=fp.name,
                    model=FakeInferenceModel(),
                ),
            )

    def test_add_label_to_prediction_result(self):
        data = np.random.randint(1, 100, len(FakeLabelData.labels)).astype(np.float32)
        data /= data.sum()

        result = messagebus.handle(
            commands.LabelPrediction(prediction=data, label_data=FakeLabelData(), topk=2)
        )
        self.assertEqual(len(result), 2)

    def test_get_labels_with_prediction(self):
        result = messagebus.handle(
            commands.MakePrediction(
                image_path=os.path.join(ROOT_DIR, 'tests', 'data', 'sample.jpg'),
                model=FakeInferenceModel(),
                topk=3,
                label_data=FakeLabelData(),
            )
        )

        self.assertEqual(len(result), 3)

    def test_raise_error_if_topk_is_negative(self):
        with self.assertRaises(ValueError):
            messagebus.handle(
                commands.MakePrediction(
                    image_path=os.path.join(ROOT_DIR, 'tests', 'data', 'sample.jpg'),
                    model=FakeInferenceModel(),
                    topk=-3,
                    label_data=FakeLabelData(),
                )
            )

    def test_raise_if_label_data_is_not_provided(self):
        with self.assertRaises(InvalidMessage):
            messagebus.handle(
                commands.MakePrediction(
                    image_path=os.path.join(ROOT_DIR, 'tests', 'data', 'sample.jpg'),
                    model=FakeInferenceModel(),
                    topk=-3,
                )
            )


if __name__ == '__main__':
    unittest.main()
