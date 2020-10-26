#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import warnings
import argparse

from flask import Flask
from werkzeug.routing import BaseConverter

from image_app.settings import set_config, get_config, ROOT_DIR
from image_app.exception import ConfigAlreadySetError
from image_app.orm.db import setup_session, init_db, session_removal


logger = logging.getLogger(__file__)


# https://stackoverflow.com/questions/5870188/does-flask-support-regular-expressions-in-its-url-routing
class RegexConverter(BaseConverter):
    def __init__(self, url_map, *items):
        super(RegexConverter, self).__init__(url_map)
        self.regex = items[0]


def create_app(mode: str) -> Flask:
    """Create flask app."""
    try:
        set_config(mode)
    except ConfigAlreadySetError:
        # if already set config, reuse it
        pass

    config = get_config()

    app = Flask(__name__,
                static_folder=config.STATIC_DIR,
                template_folder=config.TEMPLATE_DIR)

    if not os.path.exists(os.path.join(ROOT_DIR, config.STATIC_DIR, config.UPLOAD_DIR)):
        os.makedirs(os.path.join(ROOT_DIR, config.STATIC_DIR, config.UPLOAD_DIR))
    app.config.from_object(config)
    app.url_map.converters['regex'] = RegexConverter

    setup_session(config)

    from image_app.web import base as base_bp
    app.register_blueprint(base_bp)

    from image_app.web.api import api as api_bp
    app.register_blueprint(api_bp)

    if config.DEBUG:
        with app.app_context():
            init_db()

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        session_removal()

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--mode', type=str,
                        default='dev',
                        help='Configuration type')

    args = parser.parse_args()

    mode = args.mode
    if mode == 'production':
        warnings.warn('Production mode should not be run from `app.py`')

    elif mode not in ('test', 'dev'):
        mode = 'dev'

    from image_app.settings import set_config, get_config
    set_config(mode)

    app = create_app(get_config())
    app.run()
