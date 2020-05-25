#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import glob
import argparse

import numpy as np
import tensorflow as tf

from main.settings import ROOT_DIR
from main.ml.models import BaseModel, SimpleModel
from main.ml.data import load_tfrecord_files, data_generator

_acc = tf.keras.metrics.Accuracy()


# only work in eager mode due to converting tensor to numpy array
def calculate_accuracy(pred, labels):
    if len(pred.shape) > 1:
        pred = tf.argmax(pred, 1, output_type=tf.int32)

    correct = tf.equal(pred, tf.cast(labels, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()
    correct = tf.reduce_sum(tf.cast(correct, tf.int32)).numpy()
    return accuracy, correct


class TrainModel(object):

    def __init__(self, batch_size):
        #self.model = BaseModel(num_classes=121)
        self.model = SimpleModel(n_classes=121, dropout=0.0)
        self.batch_size = batch_size
        # TODO: add choices
        self.optimizer = tf.keras.optimizers.SGD()

    #@tf.function
    def _train(self, inputs, labels):

        with tf.GradientTape() as tape:
            out = self.model(inputs, training=True)
            cost = tf.keras.losses.sparse_categorical_crossentropy(labels, out, from_logits=False)
            loss = tf.reduce_mean(cost)

        trainables = self.model.trainable_variables
        gradients = tape.gradient(loss, trainables)
        self.optimizer.apply_gradients(zip(gradients, trainables))

        accuracy, correct = calculate_accuracy(out, labels)

        return loss, accuracy, correct

    def train(self, epochs=10, iter_per_epoch=100, verbose_step=100):
        start = time.perf_counter()

        print_format = '{title} - Loss: {loss:.4f} - Acc: {acc:.4f} - Time: {time:.2f}s'

        train_dataset = glob.glob(os.path.join(ROOT_DIR, 'data/train/file_??.tfrecord')) + \
            glob.glob(os.path.join(ROOT_DIR, 'data/train/file_aug*.tfrecord'))

        best_loss = 5.0
        # consective count if not improved in validation
        persist_count = 0

        gen = data_generator(train_dataset, buffer_size=2000, repeat=True, batch_size=self.batch_size)

        for epoch in range(1, epochs + 1):
            start_epoch = time.perf_counter()
            total_loss = 0
            total_correct = 0
            size = 0
            print(f'Epoch: {epoch}')

            for i in range(1, iter_per_epoch + 1):
                st = time.perf_counter()
                x_train, y_train = next(gen)
                loss, acc, correct = self._train(x_train, y_train)
                size += self.batch_size
                total_loss += loss * self.batch_size
                total_correct += correct

                if i % verbose_step == 0:
                    print(print_format.format(
                        title=f'  Iter {i}',
                        loss=total_loss/size,
                        acc=total_correct/size,
                        time=time.perf_counter() - st))

            total_loss /= size
            total_acc = total_correct / size

            print(f'Epoch {epoch} - Total loss: {total_loss:.4f}  Accuracy: {total_acc:.4f}')

            val_loss = 0
            val_acc = 0
            size = 0
            st = time.perf_counter()
            total_correct = 0

            val_dataset = glob.glob(os.path.join(ROOT_DIR, 'data/test/*.tfrecord'))

            for i, (x_val, y_val) in enumerate(data_generator(val_dataset, shuffle=False, batch_size=32), 1):
                pred = self.model.predict(x_val)
                cost = tf.keras.losses.sparse_categorical_crossentropy(y_val, pred, from_logits=True)
                val_loss += tf.reduce_sum(cost)
                total_correct += calculate_accuracy(pred, y_val)[1]
                size += x_val.shape[0]

            val_loss /= size
            val_acc = total_correct / size

            print(print_format.format(title='Validation', loss=val_loss, acc=val_acc, time=time.perf_counter() - st))

            if val_loss < best_loss:
                print(f'Lowest loss {best_loss:.3f} => {val_loss:.3f}: saving model...')
                best_loss = val_loss
                self.model.save_weights(os.path.join(ROOT_DIR, 'saved_model'), save_format='tf')
                persist_count = 0
            else:
                persist_count += 1

            epoch_total_time = time.perf_counter() - start_epoch
            print(f'Total time per epoch: {epoch_total_time} sec.')

        total_time = int(time.perf_counter() - start)
        print(f'Total training time: {total_time} sec.')

        if persist_count == 5:
            print('Not updated in 5 times. Early stopping...')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs in training step')
    parser.add_argument('--iter-per-epoch', type=int, default=100,
                        help='Number of iteration to run in each epoch')
    parser.add_argument('--verbose-step', type=int, default=100,
                        help='Number of steps for each display')

    args = parser.parse_args()

    train_model = TrainModel(args.batch)
    train_model.train(epochs=args.epochs,
                      iter_per_epoch=args.iter_per_epoch,
                      verbose_step=args.verbose_step)
