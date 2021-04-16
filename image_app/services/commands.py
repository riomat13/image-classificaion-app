#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
from werkzeug.datastructures import FileStorage

from image_app.ml.base import LabelData
from image_app.ml.infer import InferenceModel


class Command:
    pass


@dataclass
class UploadImage(Command):
    file_object: FileStorage


@dataclass
class MakePrediction(Command):
    image_path: str
    model: InferenceModel


@dataclass
class LabelPrediction(Command):
    prediction: np.ndarray
    label_data: LabelData
    topk: int
