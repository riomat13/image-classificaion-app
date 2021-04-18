#!/usr/bin/env python3

import os

from .base import LabelData, DogBreedClassificationLabelData
from .infer import InferenceModel, DogBreedClassificationInferenceModel


__all__ = ('ML_MODELS', 'MODEL_TYPES', 'InferenceModel', 'LabelData', 'get_model_data', 'get_label_data')

# ML Model settings
ML_MODELS = {
    'DOG_BREED': {
        'MODEL_DATA': DogBreedClassificationInferenceModel(),
        'LABEL_DATA': DogBreedClassificationLabelData()
    },
}

MODEL_TYPES = {'dog_breed'}


def get_model_data(model_type: str) -> InferenceModel:
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Invalid model type: {model_type}')

    return ML_MODELS[model_type.upper()]['MODEL_DATA']


def get_label_dataset(model_type: str) -> LabelData:
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Invalid model type: {model_type}')

    return ML_MODELS[model_type.upper()]['LABEL_DATA']
