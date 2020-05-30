#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import sqlalchemy

from image_app.orm.db import provide_session, get_session


logger = logging.getLogger(__file__)


class _QueryProperty(object):

    def __init__(self):
        self.sess = None

    def __get__(self, instance, instance_type):
        if self.sess is None:
            self.sess = get_session()
        if instance is None:
            return self.sess.query(instance_type)
        return self.sess.query(type(instance))


class BaseModelMixin(object):

    query = _QueryProperty()

    @classmethod
    def get(cls, id):
        return cls.query.get(id)

    @provide_session
    def save(self, *, autoflush=True, session=None):  # pragma: no cover
        try:
            session.add(self)
            if autoflush:
                session.flush()
        except sqlalchemy.exc.IntegrityError as e:
            logger.error(e)
            session.rollback()
        except Exception as e:
            logger.error(e)
            session.rollback()
            raise

    @provide_session
    def delete(self, *, session=None):  # pragma: no cover
        session.delete(self)
