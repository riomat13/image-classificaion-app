#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BaseError(Exception):
    pass


class ConfigAlreadySetError(BaseError):
    """Raise when config is already set."""
    pass
