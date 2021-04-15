#!/usr/bin/env python3

from typing import List, Tuple

import numpy as np
from PIL import UnidentifiedImageError

from . import commands
from image_app.exception import InvalidImageDataFormat
from image_app.types import PredictionResult
from image_app.ml.base import LabelData
from image_app.ml.data import convert_image_to_tensor
from image_app.ml.infer import InferenceModel
from image_app.ml.preprocess import load_image, preprocess_input


def make_prediction(cmd: commands.MakePrediction) -> np.ndarray:
    """Make prediction from image.

    Args:
        cmd: commands.MakePrediction
            prediction command

    Returns:
        np.ndarray: result of the prediction

    Raises:
        FileNotFoundError: If given file path does not exist
        InvalidImageDataFormat: If content of a file is not an image
    """
    try:
        image = load_image(cmd.image_path)
    except UnidentifiedImageError as e:
        raise InvalidImageDataFormat(f'The file data is not readable: {cmd.image_path}')
    image = preprocess_input(image)
    return cmd.model.infer(image)


def label_prediction(cmd: commands.LabelPrediction) -> List[PredictionResult]:
    """Add label to the predicted result."""
    most_likelies = np.argsort(cmd.prediction)[-cmd.topk:]
    return [(cmd.label_data.get_label_by_id(most_likelies[i]), cmd.prediction[most_likelies[i]])
            for i in reversed(range(cmd.topk))]
