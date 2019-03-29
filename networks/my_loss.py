# -*- coding: utf-8 -*-
import tensorflow as tf

def WALF_loss(target, output, alpha=[0.2, 0.3, 0.5]):
    output /= tf.reduce_sum(output, axis=-1, keep_dims=True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(10e-8, output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum(alpha * (1. - output) * target * tf.log(output),
                           axis=-1)

def focal_loss(target, output, gamma=2, alpha=0.5):
    output /= tf.reduce_sum(output, axis=-1, keep_dims=True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(10e-8, output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum(alpha * (1. - output) ** gamma * target * tf.log(output),
                           axis=-1)


def W_loss(target, output, alpha=[0.2, 0.3, 0.5]):
    output /= tf.reduce_sum(output, axis=-1, keep_dims=True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(10e-8, output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum(alpha * target * tf.log(output),
                           axis=-1)
