#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='image-app',
    version='0.0.1dev0',
    packages=find_packages(exclude=['tests']),
    entry_points='''
        [console_scripts]
        image-app=image_app.cli:cli
        image-app-pre=image_app.preprocess_raw_data:main
    '''
)
