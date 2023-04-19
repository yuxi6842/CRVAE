#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi
#
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras

from tensorflow.compat.v1.keras import backend as K
import sys; sys.path.append('.')
from tensorflow.compat.v1.keras import activations, initializers, regularizers, constraints
from tensorflow.compat.v1.keras.layers import Layer, InputSpec
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class ComplexDense(Layer):
    """Regular complex densely-connected NN layer.
    `Dense` implements the operation:
    `real_preact = dot(real_input, real_kernel) - dot(imag_input, imag_kernel)`
    `imag_preact = dot(real_input, imag_kernel) + dot(imag_input, real_kernel)`
    `output = activation(K.concatenate([real_preact, imag_preact]) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    AN ERROR MESSAGE IS PRINTED.
    # Arguments
        units: Positive integer, dimensionality of each of the real part
            and the imaginary part. It is actualy the number of complex units.
        activation: Activation function to use
            (see keras.activations).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the complex `kernel` weights matrix.
            By default it is 'complex'.
            and the usual initializers could also be used.
            (see keras.initializers and init.py).
        bias_initializer: Initializer for the bias vector
            (see keras.initializers).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see keras.regularizers).
        bias_regularizer: Regularizer function applied to the bias vector
            (see keras.regularizers).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see keras.regularizers).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see keras.constraints).
        bias_constraint: Constraint function applied to the bias vector
            (see keras.constraints).
    # Input shape
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        For a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='he',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        # tf.keras.backend.set_floatx('float64')
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        # self.input_spec = InputSpec(ndim=2)
        # self.supports_masking = True

    def build(self, input_shape):
        # assert len(input_shape) == 2
        # assert input_shape[-1] % 2 == 0
        # tf.keras.backend.set_floatx('float64')

        input_dim = input_shape[-1]
        tf.get_seed(self.seed)
        kernel_shape = (input_dim, self.units)
        if self.init_criterion == 'he':
            fan_in = kernel_shape[-2]
            s = np.sqrt(2. / fan_in, dtype=np.float32)

            # initializer = tf.keras.initializers.he_normal()
            # s = initializer(shape=kernel_shape)
        elif self.init_criterion == 'glorot':
            initializer = tf.keras.initializers.glorot_normal()
            s = initializer(shape=kernel_shape)

        # Initialization using euclidean representation:
        # rng.normal
        def init_w_real(shape, dtype=None):
            return tf.random.normal(
                shape=kernel_shape,
                mean=0,
                stddev=s
            )

        def init_w_imag(shape, dtype=None):
            return tf.random.normal(
                shape=kernel_shape,
                mean=0,
                stddev=s
            )

        real_init = tf.random_normal_initializer(mean=0, stddev=s)
        imag_init = tf.random_normal_initializer(mean=0, stddev=s)

        self.real_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=real_init,
            dtype=tf.float64,
            name='real_kernel'
        )
        self.imag_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=imag_init,
            dtype=tf.float64,
            name='imag_kernel'
        )

        print('real_kernel ', self.real_kernel)
        print('imag_kernel ', self.imag_kernel)

        if self.use_bias:
            self.rbias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=tf.float64
            )
            self.ibias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=tf.float64
            )
        else:
            self.bias = None

        # self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs):
        # tf.keras.backend.set_floatx('float64')

        # input_shape = K.shape(inputs)
        # input_dim = input_shape[-1] // 2
        # real_input = inputs[:, :input_dim]
        # imag_input = inputs[:, input_dim:]

        self.w = tf.complex(self.real_kernel, self.imag_kernel)
        self.bias = tf.complex(self.rbias, self.ibias)
        print('w', self.w)
        print('inputs ', inputs)
        print('bias', self.bias)
        output = tf.matmul(inputs, self.w) + self.bias

        # output = K.dot(inputs, cat_kernels_4_complex)
        #
        # if self.use_bias:
        #     output = K.bias_add(output, self.bias)
        # if self.activation is not None:
        #     output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        # output_shape[-1] = 2 * self.units
        return tuple(output_shape)

    def get_config(self):
        # tf.keras.backend.set_floatx('float64')

        if self.kernel_initializer in {'complex'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': ki,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

