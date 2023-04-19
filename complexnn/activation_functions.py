#!/usr/bin/env python
# -*- coding: utf-8 -*-

# original code link: https://github.com/ChihebTrabelsi/deep_complex_networks
# Authors: Dmitriy Serdyuk, Olexa Bilaniuk, Chiheb Trabelsi
from tensorflow import keras

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda
import tensorflow.compat.v1 as tf


def mod_relu(z, scope='', reuse=None, dtype=tf.float64):
    """
        Implementation of the modRelu from Arjovski et al.
        f(z) = relu(|z| + b)(z / |z|) or
        f(r,theta) = relu(r + b)e^(i*theta)
        b is initialized to zero, this leads to a network, which
        is linear during early optimization.
    Input:
        z: complex input.
        b: 'dead' zone radius.
    Returns:
        z_out: complex output.

    Reference:
         Arjovsky et al. Unitary Evolution Recurrent Neural Networks
         https://arxiv.org/abs/1511.06464
    """

    with tf.variable_scope('mod_relu' + scope, reuse=reuse):
        b = tf.get_variable('b', [], dtype=dtype,
                            initializer=tf.random_uniform_initializer(-0.01, 0.01))
        modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
        rescale = tf.nn.relu(modulus + b) / (modulus + 1e-6)
        rescale = tf.complex(rescale, tf.zeros_like(rescale, dtype=dtype))
        return tf.multiply(rescale, z)

def split_relu(z, scope='', reuse=None):
    """ A split relu applying relus on the real and
        imaginary parts separately following
        Trabelsi https://arxiv.org/abs/1705.09792"""
    with tf.variable_scope('split_relu' + scope):
        x = tf.real(z)
        y = tf.imag(z)
        return tf.complex(tf.nn.relu(x), tf.nn.relu(y))

def z_relu(z, scope='', reuse=None):
    """
    The z-relu, which is active only
    in the first quadrant.
    As proposed by Guberman:
    https://arxiv.org/abs/1602.09046
    """
    with tf.variable_scope('z_relu'):
        factor1 = tf.cast(tf.real(z) > 0, tf.float32)
        factor2 = tf.cast(tf.imag(z) > 0, tf.float32)
        combined = factor1*factor2
        rescale = tf.complex(combined, tf.zeros_like(combined))
        return tf.multiply(rescale, z)

def hirose(z, scope='', reuse=None):
    """
    Compute the non-linearity proposed by Hirose.
    See for example:
    Complex Valued nonlinear Adaptive Filters
    Mandic and Su Lee Goh
    Chapter 4.3.1 (Amplitude-Phase split complex approach)
    """
    with tf.variable_scope('hirose' + scope, reuse=reuse):
        m = tf.get_variable('m', [], tf.float64,
                            initializer=tf.random_uniform_initializer(0.9, 1.1))
        modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
        # use m*m to enforce positive m.
        rescale = tf.complex(tf.nn.tanh(modulus/(m*m))/modulus,
                             tf.zeros_like(modulus))
        return tf.multiply(rescale, z)

def mod_sigmoid(z, scope='', reuse=None, dtype=tf.float64):
    """
    ModSigmoid implementation, using a coupled alpha and beta.
    """
    with tf.variable_scope('mod_sigmoid_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=dtype,
                                initializer=tf.constant_initializer(0.0))
        alpha_norm = tf.nn.sigmoid(alpha)
        pre_act = alpha_norm * tf.real(z) + (1 - alpha_norm)*tf.imag(z)
        return tf.complex(tf.nn.sigmoid(pre_act), tf.zeros_like(pre_act))

def mod_sigmoid_beta(z, scope='', reuse=None):
    """
    ModSigmoid implementation. Alpha and beta and beta are uncoupled
    and constrained to (0, 1).
    """
    with tf.variable_scope('mod_sigmoid_beta_' + scope, reuse=reuse):
        alpha = tf.get_variable('alpha', [], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        beta = tf.get_variable('beta', [], dtype=tf.float32,
                               initializer=tf.constant_initializer(1.0))
        alpha_norm = tf.nn.sigmoid(alpha)
        beta_norm = tf.nn.sigmoid(beta)
        pre_act = alpha_norm * tf.real(z) + beta_norm*tf.imag(z)
        return tf.complex(tf.nn.sigmoid(pre_act), tf.zeros_like(pre_act))






