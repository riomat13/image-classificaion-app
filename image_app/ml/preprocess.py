#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Codes in this script are based on keras_preprocessing:
#   https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/utils.py#L79


import io

import numpy as np
from PIL import Image


_PIL_INTERPOLATION_METHODS = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
}
# These methods were only introduced in version 3.4.0 (2016).
if hasattr(Image, 'HAMMING'):
    _PIL_INTERPOLATION_METHODS['hamming'] = Image.HAMMING
if hasattr(Image, 'BOX'):
    _PIL_INTERPOLATION_METHODS['box'] = Image.BOX
# This method is new in version 1.1.3 (2013).
if hasattr(Image, 'LANCZOS'):
    _PIL_INTERPOLATION_METHODS['lanczos'] = Image.LANCZOS


def _reformat_image(img, target_size, interpolation):
    """Reformat image

    Args:
        img: PIL image instance
        target_size: size after resized `(img_height, img_width)`
        interpolation: str
            Interpolation method to apply to the image during resizing

    Returns:
        numpy array (np.float32)
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return np.array(img, dtype=np.float32)


def load_image_from_array(arr, target_size=(224, 224), interpolation='nearest'):
    """Loads an image from numpy array.
    All images will be converted to `RGB` mode.

    Args:
        arr: numpy array represents image
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: str
            Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", "bicubic",
            "lanczos", "box" and "hamming".
            Default: "nearest".

    Returns:
        numpy array (np.float32)

    Raises:
        ValueError: raise when given iterpolation method name is not valid
    """
    img = Image.fromarray(arr)
    return _reformat_image(img, target_size, interpolation)


def load_image(path, target_size=(224, 224), interpolation='nearest'):
    """Loads an image.
    All images will be converted to `RGB` mode.

    Args:
        path: Path to image file.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: str
            Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", "bicubic",
            "lanczos", "box" and "hamming".
            Default: "nearest".

    Returns:
        numpy array

    Raises:
        ValueError: raise when given iterpolation method name is not valid
    """
    with open(path, 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        return _reformat_image(img, target_size, interpolation)


def preprocess_input(x):  # pragma: no code
    """Preprocesses a Numpy array encoding an image."""
    x /= 127.5
    x -= 1.
    return x
