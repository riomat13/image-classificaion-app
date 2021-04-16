#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from image_app.settings import set_config
set_config('test')

import os
import tempfile

import numpy as np

from image_app.exception import InvalidImageDataFormat
from image_app.services import commands
from image_app.services.handlers import make_prediction, label_prediction
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
        pred = make_prediction(
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
            make_prediction(
                commands.MakePrediction(
                    image_path=fp.name,
                    model=FakeInferenceModel(),
                ),
            )

    def test_add_label_to_prediction_result(self):
        data = np.random.randint(1, 100, len(FakeLabelData.labels)).astype(np.float32)
        data /= data.sum()

        result = label_prediction(
            commands.LabelPrediction(prediction=data, label_data=FakeLabelData(), topk=2)
        )
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
