'''
    1, load dataset of boston_housing

    2, set up model using function_API
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
    # function API
    x_train = data[0]
    input = keras.layers.Input(shape=x_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation='relu')(input)
    hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
    concat = keras.layers.concatenate([input, hidden2])
    output = keras.layers.Dense(1)(concat)

    # fix the model
    model = keras.models.Model(inputs = [input],
                               outputs = [output])
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