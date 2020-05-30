#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click

from image_app.app import create_app
from image_app.orm.db import (
    init_db as _init_db,
    drop_db as _drop_db,
    reset_db as _reset_db
)


@click.group()
@click.option('-c', '--config', default='dev', show_default=True)
@click.pass_context
def cli(ctx, config):
    click.echo(f'Configuration: {config}')
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.group()
def db():
    pass


@db.command()
@click.pass_context
def initdb(ctx):
    click.echo('Initializing Database')

    app = create_app(ctx.obj['config'])
    with app.app_context():
        _init_db()


@db.command()
@click.pass_context
def dropdb(ctx):
    click.echo('Dropping Database')

    app = create_app(ctx.obj['config'])
    with app.app_context():
        _drop_db()


@db.command()
@click.pass_context
def resetdb(ctx):
    click.echo('Resetting Database')

    app = create_app(ctx.obj['config'])
    with app.app_context():
        _reset_db()


if __name__ == '__main__':
    cli()
