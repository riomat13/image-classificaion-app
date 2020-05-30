#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
from functools import wraps

from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

from image_app.settings import get_config


__all__ = ['setup_session', 'get_engine', 'session_scope',
           'provide_session', 'session_removal',
           'init_db', 'drop_db', 'reset_db',
           'get_table_names', 'get_columns',
           'ModelBase']


class _Session(object):

    def __init__(self):
        self.engine = None
        self.Session = None
        self.Base = declarative_base()
        self.insp = None

    def setup_session(self, config):
        """Set up session to database."""
        self.engine = create_engine(config.DATABASE_URI)
        self.insp = inspect(self.engine)

        self.Session = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False)
        )

    def get_engine(self):
        """Get current bind engine."""
        return self.engine

    def session_removal(self):
        """Clean up session."""
        if self.Session is not None:
            self.Session.remove()
            self.Session = None

    def get_session(self):
        if self.Session is None:
            config = get_config()
            self.setup_session(config)
        return self.Session()

    @contextlib.contextmanager
    def session_scope(self):
        """Context manager to use session.
        If `setup_session()` is not called beforehand,
        current config will be used to build session.
        """
        if self.Session is None:
            config = get_config()
            self.setup_session(config)

        session = self.Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def provide_session(self, func):
        """Execute under session.
        If session is provided to the wrapped function,
        this won't do anything.
        If session is not provided,
        generate session and execute the function.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_session = 'session'

            func_params = func.__code__.co_varnames

            # check if 'session' is in args
            session_in_args = arg_session in func_params and \
                func_params.index(arg_session) < len(args)

            session_in_kwargs = arg_session in kwargs

            if session_in_args or session_in_kwargs:
                return func(*args, **kwargs)
            else:
                with self.session_scope() as sess:
                    kwargs[arg_session] = sess
                    return func(*args, **kwargs)

        return wrapper

    def init_db(self):
        self.Base.metadata.create_all(self.engine)

    def drop_db(self, conn=None):
        conn = conn or self.engine
        self.Base.metadata.drop_all(self.engine)

    def reset_db(self, conn=None):
        self.drop_db(conn)
        self.init_db()

    def get_table_names(self, schema=None):
        """Return all table names in referred to within a particular schema.
        The names are expected to be real tables only, not views.

        This is a wrapper of
            `sqlalchemy.engine.reflection.Inspector.get_table_names`
        """
        if self.insp is None:
            raise
        
        return self.insp.get_table_names(schema)

    def get_columns(self, table_name, schema=None):
        """Return information about columns in table_name.

        This is a wrapper of
            `sqlalchemy.engine.reflection.Inspector.get_columns`
        """
        if self.insp is None:
            raise

        return self.insp.get_columns(table_name, schema)


__inst = _Session()
ModelBase = __inst.Base
setup_session = __inst.setup_session
get_engine = __inst.get_engine
session_scope = __inst.session_scope
get_session = __inst.get_session
provide_session = __inst.provide_session
session_removal = __inst.session_removal
init_db = __inst.init_db
reset_db = __inst.reset_db
get_table_names = __inst.get_table_names
get_columns = __inst.get_columns
drop_db = __inst.drop_db
