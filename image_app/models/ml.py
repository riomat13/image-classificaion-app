#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from sqlalchemy import Column, String, Integer, DateTime

from image_app.orm.db import ModelBase
from image_app.models.mixins import BaseModelMixin


class PredictionLog(BaseModelMixin, ModelBase):
    __tablename__ = 'prediction_log'

    id = Column(Integer, primary_key=True)
    content = Column(String, nullable=False)
    logged_at = Column(DateTime, default=datetime.utcnow())
