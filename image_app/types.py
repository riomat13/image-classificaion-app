#!/usr/bin/env python3

from typing import NewType, Tuple


# prediction result with class ID and associated the score
PredictionResult = NewType('PredictionResult', Tuple[int, float])
