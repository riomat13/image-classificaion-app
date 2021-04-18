#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
from typing import List


class InferenceModel(abc.ABC):

    @abc.abstractmethod
    def infer(self, img):
        raise NotImplementedError


class LabelData(abc.ABC):

    @abc.abstractclassmethod
    def get_label_list(self) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    @abc.abstractclassmethod
    def get_label_by_id(self, label_id) -> str:  # pragma: no cover
        raise NotImplementedError

    @abc.abstractclassmethod
    def get_id_by_label(self, label) -> int:  # pragma: no cover
        raise NotImplementedError

    @abc.abstractstaticmethod
    def get_label_data():
        raise NotImplementedError
