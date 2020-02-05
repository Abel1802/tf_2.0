'''
    tensorflow data basic API
'''
import tensorflow as tf
import numpy as np
from tensorflow import keras

# data from numpy
print('from numpy:'.center(100, '-'))
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
for item in dataset:
    print(item)

# 1, epoch repeat,
# 2, get batch
print('repeat ecpoch, get batch:'.center(100, '-'))
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)