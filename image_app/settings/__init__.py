#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os as _os
import logging as _logging
import warnings as _warnings
import configparser as _configparser

from ._base import ROOT_DIR, Config
from ._helper import ini_parser, dotenv_parser
from image_app.exception import ConfigAlreadySetError


__all__ = ['set_config', 'set_from_envfile', 'get_config', 'ROOT_DIR',
           'ProductionConfig', 'DevelopmentConfig', 'TestConfig']


_logging.basicConfig(
    level=_logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


class _ConfigFactory(object):
    __instance = None
    __config = None

    def __new__(cls, *args):
        # this will not return instance to prevent instantiated
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args)
        else:
            raise ConfigAlreadySetError('Cannot reset config in the same process')

    def __init__(self):
        raise RuntimeError('This should not be instantiated')

    @classmethod
    def set_config(cls, config_type: str):
        """Set configuration."""
        cls.__new__(cls)

        if config_type == 'dev':
            from ._development import DevelopmentConfig
            cls.__config = DevelopmentConfig
        elif config_type == 'test':
            from ._test import TestConfig
            cls.__config = TestConfig
        elif config_type == 'production':
            from ._production import ProductionConfig
            cls.__config = ProductionConfig
        else:
            _warnings.warn('Unrecognized config type. Use `development` mode')
            cls.__config = DevelopmentConfig

    @classmethod
    def set_from_envfile(cls, path, **kwargs):
        """Set configuration from environment file.
        Parameters will be used from `image_app.settings._base.Config`
        if not exist in the env file.

        Usage:
            .ini file:
                ```example.ini
                [DEFAULT]
                DATABASE_URI=sqlite:///
                ```
                and execute,
                >>> set_from_env('example.ini', section='DEFAULT')

            other text file
                ```.env
                DATABASE_URI=sqlite:///
                ```
                and execute,
                >>> set_from_env('.env')
        """
        cls.__new__(cls)

        if path.endswith('.ini'):
            section = kwargs.get('section', 'DEFAULT')
            config = _helper.ini_parser(path, section)

        else:
            config = _helper.dotenv_parser(path)

        cls.__config = _helper.update_config(config)

    @classmethod
    def get_config(cls):
        """Return current configuration."""
        if cls.__instance is None:
            _warnings.warn('Config is not set. Returning base config')
            return Config
        return cls.__config


set_config = _ConfigFactory.set_config
set_from_envfile = _ConfigFactory.set_from_envfile
get_config = _ConfigFactory.get_config
