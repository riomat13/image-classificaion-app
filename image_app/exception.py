#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BaseError(Exception):
    pass


class ConfigAlreadySetError(BaseError):
    """Raise when config is already set."""


class FileEmptyError(BaseError):
    """Raise when a file is empty."""


class InvalidImageDataFormat(BaseError):
    """Raise when failed to load data due to wrong format."""


class InvalidMessage(BaseError):
    """Raise when message data contains invalid data."""
