#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path

from image_app.settings import ROOT_DIR


class LabelData(object):
    __instance = None

    def __init__(self):
        if LabelData.__instance is not None:
            raise RuntimeError('Already created. Use `get_label_data()` instead.')

        self._label2id = {}
        self._id2label = ['others']

        with open(os.path.join(ROOT_DIR, 'data/category_list.txt'), 'r') as f:
            for name in f:
                name = name.strip()

                # labels are assigned from `1` and all others are `0`
                self._label2id[name] = len(self._id2label)
                self._id2label.append(name)

        LabelData.__instance = self

    def __len__(self):
        return len(self._id2label)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._label2id.get(key, 0)
        elif isinstance(key, int):
            return self._id2label[key]
        elif hasattr(key, '__iter__'):
            return tuple(self.__getitem__(k) for k in key)

        raise TypeError('Looking up must be done by string or integer')

    def __iter__(self):
        for label in self._id2label:
            yield label

    def get_label_by_id(self, id):  # pragma: no cover
        return self._id2label[id]

    def get_id_by_label(self, label):  # pragma: no cover
        return self._label2id[label]

    @property
    def id2label(self):  # pragma: no cover
        return self._id2label[:]

    @property
    def label2id(self):  # pragma: no cover
        return self._label2id.copy()

    @staticmethod
    def get_label_data():
        if LabelData.__instance is None:
            obj = LabelData()

        return LabelData.__instance

    def get_label_count(self):
        """Return number of classes."""
        return self.__len__()


