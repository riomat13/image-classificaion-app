#!/usr/bin/env python3

import io
import os
from typing import List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from . import commands
from image_app.exception import FileEmptyError, InvalidImageDataFormat
from image_app.types import PredictionResult
from image_app.ml.base import LabelData
from image_app.ml.data import convert_image_to_tensor
from image_app.ml.infer import InferenceModel
from image_app.ml.preprocess import load_image, preprocess_input
from image_app.models.image import Image as ImageModel
from image_app.settings import get_config

config = get_config()


def upload_image(cmd: commands.UploadImage):
    """Upload image data from client.

    Args:
        cmd: commands.UploadImage
            file_path: str
            file_object: werkzeug.datastructures.FileStorage

    Returns:
        str

    Raises:
        InvalidImageDataFormat: If file data format is not image
    """
    img_file = cmd.file_object

    filename = secure_filename(img_file.filename)
    fname, ext = os.path.splitext(filename)
    fname = ImageModel.query.count() + 1
    filename = f'{fname:05d}{ext}'

    filepath = os.path.join(config.UPLOAD_DIR, filename)

    img_file.save(img_path:=os.path.join(config.STATIC_DIR, filepath))

    # TODO: any other image file validation?
    try:
        image = Image.open(img_path)
    except UnidentifiedImageError as e:
        os.remove(img_path)
        raise InvalidImageDataFormat('Invalid data format')
    else:
        image.close()


    if not os.stat(img_path).st_size:
        os.remove(img_path)
        raise FileEmptyError()

    img_model = ImageModel(filename=filename)
    img_model.save()

    encode = img_model.get_encode()
    return encode


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
