#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser

from ._base import Config


def ini_parser(filepath, section):
    """Parse .ini file.

    Args:
        filepath: str
            target config file path
        section: str
            section header name to parse

    Return:
        config: dict-like object (Section object)
    """
    config = configparser.ConfigParser()
    config.read(filepath)
    return config.get(section)


def dotenv_parser(filepath):
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            try:
                key, value = line.strip().split('=')
                config[key] = value
            except Exception:
                pass
    return config


def update_config(**kwargs):
    for k, v in config.items():
        setattr(Config, k, v)
    return Config
