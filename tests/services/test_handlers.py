#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import os
import tempfile

import numpy as np

from image_app.exception import ConfigAlreadySetError
from image_app.settings import set_config
try:
    set_config('test')
except ConfigAlreadySetError:
    pass

from image_app.exception import InvalidImageDataFormat, InvalidMessage
from image_app.services import commands
from image_app.services.messagebus import messagebus
from image_app.ml import InferenceModel, LabelData, ML_MODELS, MODEL_TYPES
from image_app.settings import ROOT_DIR


class FakeLabelData(LabelData):

    labels = ['test', 'infer', 'sample', 'handler']

    def get_label_list(self):
        return self.labels

    def get_label_by_id(self, label_id):
        return self.labels[label_id]

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


# monkey-patch fake model for testing
ML_MODELS['FAKE'] = {
    'MODEL_DATA': FakeInferenceModel(),
    'LABEL_DATA': FakeLabelData()
}
MODEL_TYPES.add('fake')


class HandlersTest(unittest.TestCase):

    def test_fetch_target_label(self):
        label = messagebus.handle(commands.FetchLabels(model_type='fake', label_id=2))[0]
        self.assertEqual(label, FakeLabelData.labels[2])

    def test_fetch_all_label_items(self):
        labels = messagebus.handle(commands.FetchLabels(model_type='fake'))
        self.assertEqual(labels, FakeLabelData.labels)

    def test_fetch_labels_by_invalid_model_type(self):
        with self.assertRaises(InvalidMessage):
            messagebus.handle(commands.FetchLabels(model_type='invalid'))

    def test_run_prediction(self):
        pred = messagebus.handle(
            commands.MakePrediction(
                image_path=os.path.join(ROOT_DIR, 'tests', 'data', 'sample.jpg'),
                model_type='fake',
            )
        )
        self.assertTrue(isinstance(pred, np.ndarray))

    def test_fail_prediction_by_invalid_model_type(self):
        with self.assertRaises(InvalidMessage):
            messagebus.handle(
                commands.MakePrediction(
                    image_path=os.path.join(ROOT_DIR, 'tests', 'data', 'sample.jpg'),
                    model_type='invalid',
                )
            )

    def test_raise_error_due_to_wrong_image_format_when_predicting(self):
        fp = tempfile.NamedTemporaryFile()
        fp.write(b'non-image data')

        with self.assertRaises(InvalidImageDataFormat):
            messagebus.handle(
                commands.MakePrediction(
                    image_path=fp.name,
                    model_type='fake',
                ),
            )

    def test_add_label_to_prediction_result(self):
        data = np.random.randint(1, 100, len(FakeLabelData.labels)).astype(np.float32)
        data /= data.sum()

        result = messagebus.handle(
            commands.LabelPrediction(prediction=data, model_type='fake', topk=2)
        )
        self.assertEqual(len(result), 2)

    def test_get_labels_with_prediction(self):
        result = messagebus.handle(
            commands.MakePrediction(
                image_path=os.path.join(ROOT_DIR, 'tests', 'data', 'sample.jpg'),
                model_type='fake',
                topk=3,
            )
        )

        self.assertEqual(len(result), 3)

    def test_raise_error_if_topk_is_negative(self):
        with self.assertRaises(ValueError):
            messagebus.handle(
                commands.MakePrediction(
                    image_path=os.path.join(ROOT_DIR, 'tests', 'data', 'sample.jpg'),
                    model_type='fake',
                    topk=-3,
                )
            )


if __name__ == '__main__':
    unittest.main()
