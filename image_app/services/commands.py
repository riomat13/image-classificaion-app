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
class FetchLabels(Command):
    """Fetch all label list."""
    model_type: str
    label_id : Optional[int] = None


@dataclass
class MakePrediction(Command):
    """Make prediction.

    Args:
        image_path: str
            An image file path to make prediction.
        model_type: str
            Model type to run prediction. e.g. "dog_breed"
        topk: int (default: 0)
            Convert to label if set to non-zero.
            If this is set to zero, return all prediction scores as numpy array.
            If this value is negative, ValueError will be raised.
    """
    image_path: str
    model_type: str
    topk: Optional[int] = 0


@dataclass
class LabelPrediction(Command):
    """Convert predicted result to label."""
    prediction: np.ndarray
    model_type: str
    topk: int
