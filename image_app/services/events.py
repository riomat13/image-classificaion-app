#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List

import numpy as np

from image_app.ml.base import LabelData


class Event:
    pass


@dataclass
class Predicted(Event):
    """Convert predicted result to label."""
    prediction: np.ndarray
    label_data: LabelData
    topk: int
