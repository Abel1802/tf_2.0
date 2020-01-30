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
from sklearn.preprocessing import StandardScaler

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

    # data normalization
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(
        x_train.astype(np.float32).reshape(-1, 1)
    ).reshape(-1, 28, 28)
    x_validation_scaled = scaler.transform(
        x_validation.astype(np.float32).reshape(-1, 1)
    ).reshape(-1, 28, 28)
    x_test_scaled = scaler.transform(
        x_test.astype(np.float32).reshape(-1, 1)
    ).reshape(-1, 28, 28)
    return [x_train_scaled, y_train, x_validation_scaled, y_validation, x_test_scaled, y_test]


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
    # callbacks: Tensorboard, earlystopping, Modlecheckpoint
    logdir = './callbacks'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")

    callbacks = [
        keras.callbacks.TensorBoard(logdir),
        keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True),
        keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
    ]
    return model, callbacks


def train(data, model, callbacks):
    x_train, y_train = data[0], data[1]
    x_validation, y_validation = data[2], data[3]
    callbacks = callbacks
    # start training
    history = model.fit(x_train, y_train, epochs=5,
              validation_data=(x_validation, y_validation),
              callbacks = callbacks)
    model.save('./models/keras-sequential-model.h5')
def estimate(data):
    x_test, y_test = data[4], data[5]
    model = keras.models.load_model('./models/keras-sequential-model.h5')
    model.evaluate(x_test, y_test)


def main():
    data = data_process()
    # model, callbacks = keras_model()
    # train(data, model, callbacks)
    estimate(data)


if __name__ == "__main__":
    main()
