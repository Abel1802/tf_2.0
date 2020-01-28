import tensorflow as tf
import matplotlib as mpt
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
from tensorflow import keras


def data_process():
    # look up versions
    # print(tf.__version__)
    # print(sys.version_info)
    # for module in mpt, np, pd, sklearn, keras:
    #     print(module.__name__, module.__version__)

    # load dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

    # depart data_train into validation sets and train sets
    x_validation, x_train = x_train_all[:5000], x_train_all[5000:]
    y_validation, y_train = y_train_all[:5000], y_train_all[5000:]
    return [x_train, y_train, x_validation, y_validation, x_test, y_test]


def keras_model():
    # tf.keras.modles.Sequential()
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # compile the model
    model.compile(keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train(data, model):
    x_train, y_train = data[0], data[1]
    x_validation, y_validation = data[2], data[3]
    x_test, y_test = data[4], data[5]

    # start training
    history = model.fit(x_train, y_train, epochs=10,
              validation_data=(x_validation, y_validation))


def main():
    data = data_process()
    model = keras_model()
    train(data, model)


if __name__ == "__main__":
    main()