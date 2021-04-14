#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest


class _CalledCounter(object):
    """This is used to count function calls with Mock(MagicMock)."""
    def __init__(self):
        self._count = 0
        self._items = []
        self._kwarg_items = []

    def called(self, *args, **kwargs):
        self._count += 1
        if args is not None:
            self._items.append(args)
        if kwargs is not None:
            self._kwarg_items.append(kwargs)

    @property
    def count(self):
        return self._count

    @property
    def items(self):
        return self._items

    @property
    def kwarg_items(self):
        return self._kwarg_items
