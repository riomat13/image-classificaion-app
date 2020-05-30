#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import configparser
import json

import numpy as np
import tensorflow as tf

from image_app.settings import ROOT_DIR
from image_app.ml.base import LabelData
from image_app.ml.preprocess import preprocess_input
from image_app.ml.data import data_augmentation, load_images_all_by_batch, train_test_split, extract_image_file_paths
from image_app.ml.serializer import write_to_tfrecord


logger = logging.getLogger(__file__)


def _mkdirs(dirpath, type):
    if type not in ('train', 'test'):
        raise ValueError(f'`type` must be either "train" or "test", provided `{type}`')

    path = os.path.join(dirpath, type)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _chain_converter(chain):
    convert = {
        'flip_up_down': 'ud_flip',
        'flip_left_right': 'lr_flip'
    }
    return convert.get(chain, chain)


def _parse_float_value(conf, key, default):
    try:
        return conf.getfloat(key, default)
    except (ValueError, TypeError):
        return default


def _parse_ini(path, section):
    if not os.path.isfile(path):
        raise ValueError(f'Config file does not exist: {path}')

    conf_parser = configparser.ConfigParser(
        allow_no_value=True,
        interpolation=configparser.ExtendedInterpolation()
    )
    conf_parser.read(path)
    conf = conf_parser[section]

    chains = json.loads(conf.get('chains', []))

    kwargs = {
        'ud_flip': conf.getboolean('flip_up_down', False),
        'lr_flip': conf.getboolean('flip_left_right', False),
        'saturation': _parse_float_value(conf, 'saturation', None),
        'brightness': _parse_float_value(conf, 'brightness', None),
        'random_brightness': _parse_float_value(conf, 'random_brightness', None),
        'probability': _parse_float_value(conf, 'probability', 1.0),
        'chains_only': conf.getboolean('chains_only', False),
        'chains': [list(map(_chain_converter, chain.split(','))) for chain in chains],
        'test_ratio': conf.getfloat('test_ratio', 0.2)
    }

    logging.info(f'''Image augmentation setup
    Up down flip:       {kwargs['ud_flip']}
    Left right flip:    {kwargs['lr_flip']}
    Saturation:         {kwargs['saturation']}
    Brightness:         {kwargs['brightness']}
    Random brightness:  {kwargs['random_brightness']}
    Probability:        {kwargs['probability']}
    Chains only:        {kwargs['chains_only']}
    Chains:
        {chains}
    ''')

    return kwargs


def convert_image_to_tfrecord(dirpath,
                              depth=0,
                              out_dir=None,
                              filename='{}.tfrecord',
                              data_size=1024,
                              augmentation=False,
                              lr_flip=False,
                              ud_flip=False,
                              central_crop=None,
                              saturation=None,
                              brightness=None,
                              random_brightness=None,
                              chains_only=False,
                              chains=[],
                              probability=1.0,
                              test_ratio=0.2,
                              **kwargs):
    """Convert tensor to TFRecord file.

    Args:
        dirpath: str
            path the directory held input images
        depth: int (default: 0)
            depth from dirpath to places where images are located
            if drectory structure is following:

                    .
                    |-- image_type1
                    |       |-- image_type1_00.jpg
                    |       |-- image_type1_01.jpg
                    |               :
                    |-- image_type2
                    |       |-- image_type2_00.jpg
                    |       |-- image_type2_01.jpg
                    |               :

            depth should be `1`
        out_dir: str or None (default: None)
            target directory to save data
            if set to None, this will be same as `dirpath`
        filename: str
            file name to save as tfrecord
            default is `{type}_{number}.tfrecord`
            if filename contains `{}`,
            the data type (train or test) and ids are put there,
            otherwise, it goes right before extention.
            For instance, `filename = 'sample_{}_data.tfrecord'`,
            the first file name will be `sample_train_01_data.tfrecord`.
            (if test data, this will be `sample_test_01_data.tfrecord`.)
            if extention `.tfrecord` is not included, will be added to the file name.
        data_size: int (default: 1024)
            number of images to write into one tfrecord
        augmentation: bool (default: False)
            set to True if apply image augmentation
        ud_flip: bool
            set to True and it generates fliped image vertically
        lr_flip: bool
            set to True and it generates fliped image horiaontally
        central_crop: float
            ratio of cropping, range is (0, 1] (use `tf.image.central_crop`)
        saturation: float
            factor to multiply the saturation by
            (use `tf.image.adjust_saturation`)
        brightness: float
            (use `tf.image.adjust_brightness`)
        chains_only: bool (default: False)
            set to True if only execute chained process
        chains: list
            list of list contains name of keywards of arguments such as `ud_flip`,
            and this will chaining the process to generate the data
            It will use the same parameters.
        probability: float
            probability to execute image augmentation,
            1.0 for applying all process to all images
        test_ratio: float (default: 0.2)
            ratio of dataset saved as test set

    Returns:
        None

    Raises:
        ValueError: raise when imgs and labels sizes do not match
        ValueError: raise when `test` is not in `[0.0, 1.0]`
        ValueError: raise when data_size is not positive
    """
    if data_size < 1:
        raise ValueError('`data_size` must be positive')
    if test_ratio < 0. or test_ratio > 1.:
        raise ValueError('`test_ratio` must be within [0.0, 1.0]')

    if not filename.endswith('.tfrecord'):
        filename += '.tfrecord'

    if '{}' not in filename:
        name, ext = os.path.splitext(filename)
        filename = name + '_{}' + ext

    files = extract_image_file_paths(dirpath, depth=depth)

    if test_ratio > 0.:
        train, test = train_test_split(files, test_ratio)
    else:
        train = files
        test = None

    train_dir = _mkdirs(dirpath, 'train')
    generator = load_images_all_by_batch(train, batch_size=data_size)

    if augmentation:
        aug_idx = 0
        aug_file_count = 0
        tmp_imgs = np.empty((data_size, 224, 224, 3), dtype=np.int32)
        tmp_labels = np.empty((data_size,), dtype=np.uint8)

    for i, (imgs, labels) in enumerate(generator):
        write_to_tfrecord(imgs, labels, os.path.join(train_dir, filename.format(f'train_{i:02d}')))

        # write generated data separately to tfrecord
        if augmentation:
            aug_gen = data_augmentation(imgs, labels,
                                        lr_flip=lr_flip,
                                        ud_flip=ud_flip,
                                        brightness=brightness,
                                        saturation=saturation,
                                        random_brightness=random_brightness,
                                        chains_only=chains_only,
                                        chains=chains,
                                        probability=probability)

            for aug_img, aug_label in aug_gen:
                tmp_imgs[aug_idx, ...] = aug_img
                tmp_labels[aug_idx] = aug_label
                aug_idx += 1
                if aug_idx == data_size:
                    # safe gaude from overflow during image augmentation
                    aug_imgs = np.clip(tmp_imgs, 0, 255).astype(np.uint8)
                    write_to_tfrecord(aug_imgs,
                                      tmp_labels,
                                      os.path.join(train_dir, filename.format(f'train_aug_{aug_file_count:02d}')))
                    aug_idx = 0
                    aug_file_count += 1

    if augmentation and aug_idx > 0:
        aug_imgs = np.clip(tmp_imgs[:aug_idx], 0, 255).astype(np.uint8)
        write_to_tfrecord(aug_imgs,
                          tmp_labels[:aug_idx],
                          os.path.join(train_dir, filename.format(f'train_aug_{aug_file_count:02d}')))

    # write test data to tfrecord
    if test is not None:
        generator = load_images_all_by_batch(test, batch_size=data_size)
        test_dir = _mkdirs(dirpath, 'test')

        for i, (imgs, labels) in enumerate(generator):
            write_to_tfrecord(imgs, labels,
                              os.path.join(test_dir, filename.format(f'test_{i:02d}')))


def main():
    parser = argparse.ArgumentParser(description='Convert data into tfrecord')
    parser.add_argument('-b', '--batch', type=int, default=1024,
                        help='number of data in each tfrecord file will contain')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to directory containing input data')
    parser.add_argument('--depth', type=int, default=0,
                        help='Depath from directory where the input files are located')
    parser.add_argument('-o', '--out-dir', type=str, default=None,
                        help='Directory path to save tfrecord files. If not set, use the same as input directory')
    parser.add_argument('--conf-path', type=str, default=None,
                        help='path to ".ini" file contains params for image augmentation. This must be relative path from root.')
    parser.add_argument('--conf-section', type=str, default='DEFAULT',
                        help='Section to be parsed in ".ini" file (case sensitive)')

    args = parser.parse_args()

    augmentation = args.conf_path is not None
    conf = _parse_ini(os.path.join(ROOT_DIR, args.conf_path), args.conf_section)

    convert_image_to_tfrecord(dirpath=args.path,
                              depth=args.depth,
                              out_dir=args.out_dir or args.path,
                              data_size=args.batch,
                              augmentation=augmentation,
                              **conf)


if __name__ == '__main__':
    main()
