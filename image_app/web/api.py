#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import traceback

from flask import Blueprint, Response, jsonify, request, current_app, abort
import numpy as np

from image_app.exception import FileEmptyError
from image_app.ml.base import DogBreedClassificationLabelData
from image_app.ml.infer import DogBreedClassificationInferenceModel
from image_app.models.image import Image
from image_app.services import commands
from image_app.services.messagebus import messagebus
from image_app.settings import get_config

logger = logging.getLogger(__file__)
api = Blueprint('api', __name__, url_prefix='/api/v1')
config = get_config()


# for debug
@api.route('/ping', methods=['GET'])
def ping():  # pragma: no cover
    return 'hello world', 200


@api.route('/prediction/label/<int:id>', methods=['GET'])
def fetch_label_by_id(id):
    labels = DogBreedClassificationLabelData.get_label_data()
    label_name = labels.get_label_by_id(id)

    kwargs = {
        'status': 'success',
        'label': label_name
    }
    response = jsonify(**kwargs)
    response.status_code = 200
    return response


@api.route('/prediction/labels/list', methods=['GET'])
def fetch_label_list():
    labels = DogBreedClassificationLabelData.get_label_data()
    label_list = labels.id2label

    kwargs = {
        'status': 'success',
        'labelList': label_list
    }
    response = jsonify(**kwargs)
    response.status_code = 200
    return response


def upload_image_file_object(img_file):
    """Read image file data and save to Database if valid.
    Return error if the given file is invalid.
    """

    if img_file:
        try:
            encode = messagebus.handle(commands.UploadImage(file_object=img_file))

        except FileEmptyError as e:
            kwargs = {
                'status': 'error',
                'message': 'file is emptry'
            }
            status_code = 400

        except Exception as e:
            kwargs = {
                'status': 'error',
                'message': 'failed to save image'
            }

            if config.DEBUG:  # pragma: no cover
                logger.error(traceback.format_exc())
            else:           # pragma: no cover
                logger.error(e)

            status_code = 400
        else:
            kwargs = {
                'status': 'success',
                'imgId': encode
            }
            status_code = 201

    else:  # pragma: no cover
        if config.DEBUG:
            logger.error('Counld not find image')
        kwargs = {
            'status': 'error',
            'message': 'failed to upload image'
        }

        status_code = 400

    return kwargs, status_code


@api.route('/prediction/upload/image', methods=['POST'])
def upload_image(session=None):
    """Upload image for prediction.

    The output format is:
        {
            status: str
            imgId: str
        }
    """
    img_file = request.files['file']
    kwargs, status_code = upload_image_file_object(img_file)

    response = jsonify(kwargs)
    response.status_code = status_code

    return response


def serialize_predict_result(item):
    # display probability by percentage
    prob = item[1] * 100
    return item[0], f'{prob:.2f}'


@api.route('/prediction/serve', methods=['POST'])
def predict_image():
    """Predict image from a provided file.

    The output format is:
        {
            status: str
            imgId: str
            result: List
        }
    """
    img_file = request.files['file']
    kwargs, status_code = upload_image_file_object(img_file)

    if status_code == 201:
        try:
            model = Image.get_by_encoded_id(kwargs.get('imgId'))
            filepath = os.path.join(config.STATIC_DIR, config.UPLOAD_DIR, model.filename)

            result = messagebus.handle(
                commands.MakePrediction(
                    image_path=filepath,
                    model=DogBreedClassificationInferenceModel(),
                    topk=3,
                    label_data=DogBreedClassificationLabelData.get_label_data())
            )

            # convert float to percentages to display to be readable
            result = list(map(serialize_predict_result, result))
            kwargs['result'] = result

        except Exception as e:
            if config.DEBUG:  # pragma: no cover
                logger.error(traceback.format_exc())
            else:  # pragma: no cover
                logger.error(e)
            kwargs = {
                'status': 'error',
                'message': 'a problem occured during prediction'
            }
            status_code = 500

    response = jsonify(kwargs)
    response.status_code = status_code

    return response


@api.app_errorhandler(400)
def bad_request(e):
    response = jsonify(error='400 bad request', message=str(e))
    response.status_code = 400
    return response


@api.route('/<path:invalid_url>')
def page_not_found(invalid_url):  # pragma: no cover
    response = jsonify(error='404 page not found')
    response.status_code = 404
    return response
