import tensorflow as tf
from tensorflow import keras


# DENSE
def recurrent_dense(inp, num_units, batch_norm=False):
    x = keras.layers.Dense(num_units, activation=tf.nn.leaky_relu)(inp)
    x = keras.layers.Dense(num_units, activation=tf.nn.leaky_relu)(x) + inp
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    return x


# RNN

# CNN
def strided_down(x, filter, stride, kernel=5):
    x = keras.layers.Conv1D(filter, kernel, padding='same', strides=stride, activation=tf.nn.leaky_relu)(x)
    return x


def strided_up(x, filter, stride, kernel=5):
    x = keras.layers.Conv1DTranspose(filter, kernel, padding='same', strides=stride, activation=tf.nn.leaky_relu)(x)
    return x


def encoder_down_block(x, filter_before, filter_after, use_noise):
    x = strided_down(x, filter_before, 1)
    x = strided_down(x, filter_before, 1)
    x = strided_down(x, filter_after, 2)
    if use_noise:
        x = keras.layers.GaussianNoise(.01)(x)
    return x


def encoder_up_block(x, filters):
    x = strided_up(x, filters, 2)
    x = strided_up(x, filters, 1)
    x = strided_up(x, filters, 1)
    return x
