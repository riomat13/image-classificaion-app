#!/usr/bin/env python3

import io
import os
from typing import Deque, List, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from . import commands, events
from image_app.exception import FileEmptyError, InvalidImageDataFormat, InvalidMessage
from image_app.types import PredictionResult
from image_app.ml import get_model_data, get_label_dataset
from image_app.ml.base import LabelData
from image_app.ml.infer import InferenceModel
from image_app.ml.preprocess import load_image, preprocess_input
from image_app.models.image import Image as ImageModel
from image_app.settings import get_config

config = get_config()

Message = Union[events.Event, commands.Command]


def upload_image(cmd: commands.UploadImage, _) -> str:
    """Upload image data from client.

    Args:
        cmd: commands.UploadImage
            file_path: str
            file_object: werkzeug.datastructures.FileStorage

    Returns:
        str: encoded image ID

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
    finally:
        image.close()


    img_model = ImageModel(filename=filename)
    img_model.save()

    encode = img_model.get_encode()
    return encode


def fetch_labels(cmd: commands.FetchLabels, queue: Deque[Message]) -> List[str]:
    """Fetch all label list from specified model type."""
    try:
        label_data = get_label_dataset(cmd.model_type)
    except ValueError:
        raise InvalidMessage('Invalid model type')

    if cmd.label_id is not None:
        return [label_data.get_label_by_id(cmd.label_id)]
    return label_data.get_label_list()


def make_prediction(cmd: commands.MakePrediction, queue: Deque[Message]) -> np.ndarray:
    """Make prediction from image.

    Args:
        cmd: commands.MakePrediction
            prediction command

    Returns:
        np.ndarray: result of the prediction

    Raises:
        FileNotFoundError: If given file path does not exist
        InvalidImageDataFormat: If content of a file is not an image
        ValueError: If topk is negative
    """
    try:
        image = load_image(cmd.image_path)
    except UnidentifiedImageError as e:
        raise InvalidImageDataFormat(f'The file data is not readable: {cmd.image_path}')

    image = preprocess_input(image)

    try:
        model = get_model_data(cmd.model_type)
    except ValueError:
        raise InvalidMessage('Invalid model type')

    prediction = model.infer(image)

    if cmd.topk is not None and cmd.topk != 0:
        queue.append(
            events.Predicted(prediction=prediction,
                model_type=cmd.model_type,
                topk=cmd.topk
            )
        )

    return prediction


# currently this can be used for both event and command
def to_labels(message: Message, _) -> List[PredictionResult]:
    """Add label to the predicted result."""
    if message.topk < 1:
        raise ValueError('`topk` value must be positive')

    most_likelies = np.argsort(message.prediction)[-message.topk:]
    label_data = get_label_dataset(message.model_type)

    return [(label_data.get_label_by_id(most_likelies[i]), message.prediction[most_likelies[i]])
            for i in reversed(range(message.topk))]


COMMAND_HANDLERS = {
    commands.UploadImage: upload_image,
    commands.FetchLabels: fetch_labels,
    commands.MakePrediction: make_prediction,
    commands.LabelPrediction: to_labels,
}

EVENT_HANDLERS = {
    events.Predicted: to_labels,
}
