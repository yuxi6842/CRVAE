#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi
#


from tensorflow.keras import backend as K
import sys
sys.path.append('.')
import tensorflow.compat.v1 as tf
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np

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
                 init_criterion=None,
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 dtype=tf.complex128,
                 **kwargs):
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
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True
        self._dtype = dtype


    def build(self, input_shape):
        if self._dtype == tf.complex128:
            temp_dtype = tf.float64
        elif self._dtype == tf.complex64:
            temp_dtype = tf.float32
        input_dim = input_shape[-1]
        tf.get_seed(self.seed)
        kernel_shape = (int(input_dim), self.units)

        # real_init = tf.keras.initializers.glorot_uniform(self.seed)
        # imag_init = tf.keras.initializers.glorot_uniform(self.seed)

        val = np.math.sqrt(6. / (int(input_dim) + self.units)) * 0.5
        # val = 6. / (int(input_dim) + self.units)
        self.real_kernel = tf.Variable(
            shape=kernel_shape,
            initial_value=tf.random.uniform(kernel_shape, minval=-val, maxval=val, dtype=temp_dtype),
            name='real_kernel',
            trainable=True,
            dtype=temp_dtype
        )
        self.imag_kernel = tf.Variable(
            shape=kernel_shape,
            initial_value=tf.random.uniform(kernel_shape, minval=-val, maxval=val, dtype=temp_dtype),
            name='imag_kernel',
            trainable=True,
            dtype=temp_dtype
        )


        if self.use_bias:
            self.real_bias = tf.Variable(
                shape=(self.units,),
                initial_value=tf.zeros((self.units,), dtype=temp_dtype),
                name='real_bias',
                trainable=True,
                dtype=temp_dtype
            )
            self.imag_bias = tf.Variable(
                shape=(self.units,),
                initial_value=tf.zeros((self.units,), dtype=temp_dtype),
                name='imag_bias',
                trainable=True,
                dtype=temp_dtype
            )
        else:
            self.real_bias = None
            self.imag_bias = None

    def call(self, inputs):
        # tf.keras.backend.set_floatx('float64')
        input_shape = K.shape(inputs)
        input_dim = input_shape[-1]
        # real_input = inputs[:, :input_dim]
        # imag_input = inputs[:, input_dim:]
        
        w = tf.complex(self.real_kernel, self.imag_kernel)
        b = tf.complex(self.real_bias, self.imag_bias)

        output = tf.matmul(inputs, w) + b
        print(self.activation)
        if self.activation is not None:
            r_output = tf.math.real(output)
            r_output = self.activation(r_output)
            i_output = tf.math.imag(output)
            i_output = self.activation(i_output)
            output = tf.complex(r_output, i_output)
            # output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):

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

