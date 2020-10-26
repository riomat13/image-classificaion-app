#!/usr/bin/env python3

from flask import Blueprint

base = Blueprint('base', __name__, url_prefix='/')

from . import views
