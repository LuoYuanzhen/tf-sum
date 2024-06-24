import tensorflow as tf

import numpy as np


def positional_encoding(position, d_model):

    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(
        tf.range(position)[:, tf.newaxis],
        tf.range(d_model)[tf.newaxis, :],
        d_model
    )
    angle_rads = tf.cast(angle_rads, dtype=tf.float32).numpy()
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = tf.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = tf.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)