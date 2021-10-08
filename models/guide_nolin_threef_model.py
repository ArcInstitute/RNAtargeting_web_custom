import tensorflow as tf
#from kerastuner import HyperParameters
from tensorflow import keras

from models.layers import recurrent_dense, strided_down, encoder_down_block


def guide_nolin_threef_model(args,num_strided_down=4,kernel=3,cnn_units=32, dense_units=8, recurrent_layers=1, noise=True):
    #seq = keras.Input(shape=(30, 4))
    seq = keras.Input(shape=(args.guidelength, 4))
    other = keras.Input(shape=3)

    x = seq
    for _ in range(num_strided_down):
        x = strided_down(x, cnn_units, stride=1, kernel=3)
        if noise:
            x = keras.layers.GaussianNoise(.01)(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Concatenate()([x, other])

    x = keras.layers.Dense(dense_units, activation=tf.nn.leaky_relu)(x)
    for _ in range(recurrent_layers):
        x = recurrent_dense(x, dense_units)

    outputs = keras.layers.Dense(1)(x)
    # TODO: make a second output that is confidence, and have some allowance of reduced penalty
    #  for low confidence wrong guesses, but overall penalty for low confidence
    return keras.Model(inputs=[seq,other], outputs=outputs)
