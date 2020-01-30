'''
    1, load dataset of boston_housing

    2, set up model using subclass_API
'''
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
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def data_process():
    # load dataset
    housing = fetch_california_housing()
    # look up the imformation of this dataset
    print(housing.DESCR)
    print(housing.data.shape)
    print(housing.target.shape)

    # split datasets
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        housing.data, housing.target, random_state=7
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_all, y_train_all, random_state=11
    )
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    # data normalization
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_test_scaled = scaler.transform(x_test)

    return [x_train_scaled, y_train, x_valid_scaled, y_valid, x_test_scaled, y_test]


def keras_model(data):
    # subclass API
    class WideDeepModel(keras.models.Model):
        def __init__(self):
            super(WideDeepModel, self).__init__()
            '''定义模型层次'''
            self.hidden1_layer = keras.layers.Dense(30, activation='relu')
            self.hidden2_layer = keras.layers.Dense(30, activation='relu')
            self.output_layer = keras.layers.Dense(1)

        def call(self, input):
            '''完成模型的正向计算'''
            hidden1 = self.hidden1_layer(input)
            hidden2 = self.hidden2_layer(hidden1)
            concat = keras.layers.concatenate([input, hidden2])
            output = self.output_layer(concat)
            return output

    # fix the model
    model = WideDeepModel()
    model.build(input_shape=(None, 8))
    print(model.summary())

    # compile the model
    # model.compile(keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


def train(data, model):
    x_train, y_train = data[0], data[1]
    x_valid, y_valid = data[2], data[3]

    callbacks = [keras.callbacks.EarlyStopping(
        patience=5, min_delta=1e-2
    )]

    # start training
    history = model.fit(x_train, y_train,
                        epochs = 100,
                        validation_data=(x_valid, y_valid),
                        callbacks = callbacks
                        )

def test(data, model):
    x_test, y_test = data[4], data[5]
    model.evaluate(x_test, y_test)


def main():
    data = data_process()
    model = keras_model(data)
    train(data, model)
    test(data, model)


if __name__ == "__main__":
    main()