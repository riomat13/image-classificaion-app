#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import os.path

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from image_app.settings import get_config, ROOT_DIR
from image_app.ml.preprocess import load_image, load_image_from_array, preprocess_input


class InferenceModel(abc.ABC):

    @abc.abstractmethod
    def infer(self, img, topk=3):
        raise NotImplementedError


class DogBreedClassificationInferenceModel(InferenceModel):
    __instance = None
    __model = None

    def __init__(self):
        self._load_model()
        if DogBreedClassificationInferenceModel.__instance is None:
            DogBreedClassificationInferenceModel.__instance = self

    @classmethod
    def _load_model(cls):
        if cls.__model is None:
            config = get_config()
            model_path = os.path.join(ROOT_DIR, config.MODEL_PATH)

            interpreter = tflite.Interpreter(model_path=config.MODEL_PATH)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            input_index = input_details[0]['index']
            output_details = interpreter.get_output_details()
            output_index = output_details[0]['index']

            cls.__model = {
                'interpreter': interpreter,
                'input_index': input_index,
                'output_index': output_index
            }

    def infer(self, img, topk=3):
        """Infer image.

        Args:
            img: 3-d numpy array
                data represents a single image used for inference
            topk: int
                top K most likely labels to return

        Returns:
            np.ndarray
                prediction scores
        """
        if img.shape != (224, 224, 3):
            img = load_image_from_array(img, target_size=(224, 224))

        img = preprocess_input(img)[np.newaxis,...]

        interpreter = self.__model['interpreter']
        input_index = self.__model['input_index']
        output_index = self.__model['output_index']
        interpreter.set_tensor(input_index, img)

        interpreter.invoke()

        pred = interpreter.get_tensor(output_index)[0]
        return pred
