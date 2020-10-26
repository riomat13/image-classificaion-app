#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path

from flask import render_template, send_from_directory

from . import base

from image_app.settings import get_config


config = get_config()


@base.route('/', defaults={'path': ''})
@base.route('/<path:path>')
def index(path):  # pragma: no cover
    return render_template('index.html')


@base.route(r"/static/<regex('(.*?)\.(jpg|png|ico|js|json|txt)$'):file>", methods=["GET"])
def public(file):
    # access public directory in frontend app
    return send_from_directory(config.STATIC_DIR, file)
