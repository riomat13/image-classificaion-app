#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Boolean

try:
    from image_app.models._utils import generate_code, validate_id, _decoder, _unhash
except ModuleNotFoundError:  # pragma: no cover
    # temporary encoder
    validate_id = lambda target, x: int(target) == x
    generate_code = lambda x: str(x)
    _decoder = lambda x: x
    _unhash = lambda x: int(x)

from image_app.orm.db import ModelBase, provide_session
from image_app.models.mixins import BaseModelMixin


class Image(BaseModelMixin, ModelBase):
    __tablename__ = 'image'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False, unique=True)
    created = Column(DateTime, default=datetime.utcnow())
    category = Column(Integer)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)

    def validate_code(self, target):
        """Check given encoded string is valid."""
        return validate_id(target, self.id)

    def get_encode(self):
        return generate_code(self.id)

    def set_processed(self):
        self.processed = True
        self.processed_at = datetime.utcnow()
        self.save()

    @staticmethod
    def get_by_encoded_id(encode):
        return Image.get(_unhash(_decoder(encode)))
