#!/usr/bin/env python3

import tensorflow as tf

from image_app.ml.data import LabelData


class SimpleModel(tf.keras.Model):
    def __init__(self, n_classes, dropout=0.5):
        super(SimpleModel, self).__init__()

        self.cnn1 = tf.keras.layers.Conv2D(16, 3, strides=(1, 1), padding='valid', input_shape=(224, 224, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.cnn2 = tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.cnn3 = tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding='valid')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.cnn4 = tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding='valid')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.cnn5 = tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding='valid')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.do1 = tf.keras.layers.Dropout(dropout)
        self.do2 = tf.keras.layers.Dropout(dropout)
        self.do3 = tf.keras.layers.Dropout(dropout)
        self.do4 = tf.keras.layers.Dropout(dropout)
        self.do5 = tf.keras.layers.Dropout(dropout)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.cnn1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.do1(x, training=training)
        x = self.pool1(x)

        x = self.cnn2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.do2(x, training=training)
        x = self.pool2(x)

        x = self.cnn3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.do3(x, training=training)
        x = self.pool3(x)

        x = self.cnn4(x)
        x = self.bn4(x, training=training)
        x = tf.nn.relu(x)
        x = self.do4(x, training=training)
        x = self.pool4(x)

        x = self.cnn5(x)
        x = self.bn5(x, training=training)
        x = tf.nn.relu(x)
        x = self.do5(x, training=training)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        out = self.out(x)

        return out
