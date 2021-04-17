#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional

import numpy as np
from werkzeug.datastructures import FileStorage

from image_app.ml.base import LabelData
from image_app.ml.infer import InferenceModel


class Command:
    pass


@dataclass
class UploadImage(Command):
    """Save uploaded image to storage."""
    file_object: FileStorage


@dataclass
class MakePrediction(Command):
    """Make prediction.

    Args:
        image_path: str
            An image file path to make prediction.
        model: ml.infer.InferenceModel
            Model to run prediction.
        topk: int (default: 0)
            Convert to label if set to non-zero.
            If this is set to zero, return all prediction scores as numpy array.
            If this value is negative, ValueError will be raised.
        label_data: ml.base.LabelData (default: None)
            Used if `topk` is set as positive value
            to convert results to labels.
            This must be specified if `topk` is given.
    """
    image_path: str
    model: InferenceModel
    topk: Optional[int] = 0
    label_data: Optional[LabelData] = None


@dataclass
class LabelPrediction(Command):
    """Convert predicted result to label."""
    prediction: np.ndarray
    label_data: LabelData
    topk: int
