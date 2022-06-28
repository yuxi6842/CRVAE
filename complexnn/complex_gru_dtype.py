#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi
#
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import backend as K
import sys
sys.path.append('.')
import tensorflow.compat.v1 as tf
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
import numpy as np
import collections

from .activation_functions import mod_relu, split_relu, hirose, mod_sigmoid, mod_sigmoid_beta

_URNNStateTuple = collections.namedtuple("URNNStateTuple", ("o", "h"))
class URNNStateTuple(_URNNStateTuple):
    """Tuple used by URNN Cells for `state_size`, `zero_state`, and output state.
       Stores two elements: `(c, h)`, in that order.
       Only used when `state_is_tuple=True`.
    """
    slots__ = ()
    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


def arjovski_init(shape, dtype=tf.float64, partition_info=None):
    '''
    Use Arjovsky's unitary basis as initialization.
    Reference:
         Arjovsky et al. Unitary Evolution Recurrent Neural Networks
         https://arxiv.org/abs/1511.06464
    '''
    print("Arjosky basis initialization.")
    assert shape[0] == shape[1]
    omega1 = np.random.uniform(-np.pi, np.pi, shape[0])
    omega2 = np.random.uniform(-np.pi, np.pi, shape[0])
    omega3 = np.random.uniform(-np.pi, np.pi, shape[0])

    vr1 = np.random.uniform(-1, 1, [shape[0], 1])
    vi1 = np.random.uniform(-1, 1, [shape[0], 1])
    v1 = vr1 + 1j*vi1
    vr2 = np.random.uniform(-1, 1, [shape[0], 1])
    vi2 = np.random.uniform(-1, 1, [shape[0], 1])
    v2 = vr2 + 1j*vi2

    D1 = np.diag(np.exp(1j*omega1))
    D2 = np.diag(np.exp(1j*omega2))
    D3 = np.diag(np.exp(1j*omega3))

    vvh1 = np.matmul(v1, np.transpose(np.conj(v1)))
    beta1 = 2./np.matmul(np.transpose(np.conj(v1)), v1)
    R1 = np.eye(shape[0]) - beta1*vvh1

    vvh2 = np.matmul(v2, np.transpose(np.conj(v2)))
    beta2 = 2./np.matmul(np.transpose(np.conj(v2)), v2)
    R2 = np.eye(shape[0]) - beta2*vvh2

    perm = np.random.permutation(np.eye(shape[0], dtype=np.float32)) \
        + 1j*np.zeros(shape[0])

    fft = np.fft.fft
    ifft = np.fft.ifft

    step1 = fft(D1)
    step2 = np.matmul(R1, step1)
    step3 = np.matmul(perm, step2)
    step4 = np.matmul(D2, step3)
    step5 = ifft(step4)
    step6 = np.matmul(R2, step5)
    unitary = np.matmul(D3, step6)
    eye_test = np.matmul(np.transpose(np.conj(unitary)), unitary)
    unitary_test = np.linalg.norm(np.eye(shape[0]) - eye_test)
    print('I - Wi.H Wi', unitary_test, unitary.dtype)
    assert unitary_test < 1e-10, "Unitary initialization not unitary enough."
    stacked = np.stack([np.real(unitary), np.imag(unitary)], -1)
    assert stacked.shape == tuple(shape), "Unitary initialization shape mismatch."
    return tf.constant(stacked, dtype)


def complex_matmul(x, num_proj, scope, reuse, bias=False, bias_init_r=0.0,
                   bias_init_i=0.0, unitary=False, split_orthogonal=False,
                   unitary_init=arjovski_init, dtype=tf.float64):
    """
    Compute Ax + b.
    Arguments:
        x: A complex input vector.
        num_proj: The desired dimension of the output.
        scope: This string under which the variables will be
               registered.
        reuse: If this bool is True, the variables will be reused.
        bias: If True a bias will be added.
        bias_init_r: How to initialize the real part of the bias, defaults to zero.
        bias_init_i: How to initialize the imaginary part of the bias, defaults to zero.
        split_orthogonal: If true A's real and imaginary parts will be
                    initialized orthogonally and kept orthogonal (make sure to use the
                    Stiefel optimizer if orthogonality is desired).
        unitary: If true A will be initialized and kept in a unitary state
                 (make sure to use the Stiefel optimizer)
        unitary_init: The initialization method for the unitary matrix.
        dtype: data type [tf.float32, tf.float64]
    Returns:
        Ax + b: A vector of size [batch_size, num_proj]

    WARNING:
    Simply setting split_orthogonal or unitary to True is not enough.
    Use the Stiefel optimizer as well to enforce orthogonality/unitarity.
    """
    in_shape = tf.Tensor.get_shape(x).as_list()
    with tf.variable_scope(scope, reuse=reuse):
        if unitary:
            with tf.variable_scope('unitary_stiefel', reuse=reuse):
                varU = tf.get_variable('gate_U',
                                       shape=in_shape[-1:] + [num_proj] + [2],
                                       dtype=dtype,
                                       initializer=unitary_init)
                A = tf.complex(varU[:, :, 0], varU[:, :, 1])
        elif split_orthogonal:
            with tf.variable_scope('orthogonal_stiefel', reuse=reuse):
                Ar = tf.get_variable('gate_Ur', in_shape[-1:] + [num_proj],
                                     dtype=dtype,
                                     initializer=tf.orthogonal_initializer())
                Ai = tf.get_variable('gate_Ui', in_shape[-1:] + [num_proj],
                                     dtype=dtype,
                                     initializer=tf.orthogonal_initializer())
                A = tf.complex(Ar, Ai)
        else:
            varU = tf.get_variable('gate_A',
                                   shape=in_shape[-1:] + [num_proj] + [2],
                                   dtype=dtype,
                                   initializer=tf.glorot_uniform_initializer())
            A = tf.complex(varU[:, :, 0], varU[:, :, 1])
        if bias:
            varbr = tf.get_variable('bias_r', [num_proj], dtype=dtype,
                                    initializer=tf.constant_initializer(bias_init_r))
            varbc = tf.get_variable('bias_c', [num_proj], dtype=dtype,
                                    initializer=tf.constant_initializer(bias_init_i))
            b = tf.complex(varbr, varbc)
            return tf.matmul(x, A) + b
        else:
            return tf.matmul(x, A)

class StiefelGatedRecurrentUnit(tf.nn.rnn_cell.RNNCell):
    '''
    Implementation of a Stiefel Gated Recurrent unit.
    '''
    def __init__(self, units,
                 activation=mod_relu,
                 gate_activation=mod_sigmoid,
                 reuse=None,
                 stiefel=True,
                 single_gate=False,
                 dtype=tf.complex128):
        """
        Arguments:
            units: The size of the hidden state.
            activation: State to state non-linearity.
            gate_activation: The gating non-linearity.
            num_proj: Output dimension.
            reuse: Reuse graph weights in existing scope.
            stiefel: If True the cell will be used using the Stiefel
                     optimization scheme from Wisdom et al.
            real: If true a real valued cell will be created.
            complex_input: If true the cell expects a complex input.
            arjovski_basis: If true Arjovski et al.'s parameterization
                            is used for the state transition matrix.
        """
        super().__init__(_reuse=reuse)
        self._units = units
        self._output_size = units
        self._activation = activation
        self._stiefel = stiefel
        self._gate_activation = gate_activation
        self._single_gate = single_gate
        self._dtype = dtype

#     @tf.function
    @property
    def state_size(self):
        return URNNStateTuple(self._units, self._units)

#     @tf.function
    @property
    def output_size(self):
        return self._output_size
    
#     @tf.function
    def zero_state(self, batch_size, dtype=tf.complex128):
        first_state_dtype = tf.float32
        if dtype == tf.complex128:
            first_state_dtype = tf.float64
        first_state = tf.complex(tf.zeros([batch_size, self._units], dtype=first_state_dtype),
                                 tf.zeros([batch_size, self._units], dtype=first_state_dtype))
        if self._output_size:
            out = tf.zeros([batch_size, self._output_size], dtype=dtype)
        else:
            out = tf.zeros([batch_size, self._units*2], dtype=dtype)
        return URNNStateTuple(out, first_state)

#     @tf.function
    def double_memory_gate(self, h, x, scope, bias_init=4.0, dtype=tf.complex128):
        """
        Complex GRU gates, the idea is that gates should make use of phase information.
        """
        temp_dtype = tf.float64
        if dtype == tf.complex64:
            temp_dtype = tf.float32
        with tf.variable_scope(scope, self._reuse):
            ghr = complex_matmul(h, self._units, scope='ghr', reuse=self._reuse, dtype=temp_dtype)
            gxr = complex_matmul(x, self._units, scope='gxr', reuse=self._reuse,
                                 bias=True, bias_init_i=bias_init,
                                 bias_init_r=bias_init, dtype=temp_dtype)
            gr = ghr + gxr

            r = self._gate_activation(gr, 'r', self._reuse, dtype=temp_dtype)


            ghz = complex_matmul(h, self._units, scope='ghz', reuse=self._reuse, dtype=temp_dtype)
            gxz = complex_matmul(x, self._units, scope='gxz', reuse=self._reuse,
                                 bias=True, bias_init_i=bias_init,
                                 bias_init_r=bias_init, dtype=temp_dtype)
            gz = ghz + gxz

            z = self._gate_activation(gz, 'z', self._reuse, dtype=temp_dtype)
            return r, z

#     @tf.function
    def single_memory_gate(self, h, x, scope, bias_init, dtype=tf.complex128):
        """
        Use the real and imaginary parts of the gate equation to do the gating.
        """
        temp_dtype1 = tf.float64
        if dtype == tf.complex64:
            temp_dtype1 = tf.float32
        with tf.variable_scope(scope, self._reuse):
            ghs = complex_matmul(h, self._units, scope='ghs', reuse=self._reuse, dtype=temp_dtype1)
            gxs = complex_matmul(x, self._units, scope='gxs', reuse=self._reuse,
                                 bias=True, bias_init_i=bias_init,
                                 bias_init_r=bias_init, dtype=temp_dtype1)
            gs = ghs + gxs
            return (tf.complex(tf.nn.sigmoid(tf.real(gs)),
                               tf.zeros_like(tf.real(gs), dtype=temp_dtype1)),
                    tf.complex(tf.nn.sigmoid(tf.imag(gs)),
                               tf.zeros_like(tf.imag(gs), dtype=temp_dtype1)))

#     @tf.function
    def __call__(self, inputs, state):
        """
        Evaluate the cell equations.
        Params:
            inputs: The input values.
            state: the past cell state.
        Returns:
            output and new cell state touple.
        """
        if self._dtype == tf.complex128:
            temp_dtype_ = tf.float64
        elif self._dtype == tf.complex64:
            temp_dtype_ = tf.float32
        with tf.variable_scope("ComplexGatedRecurrentUnit", reuse=self._reuse):
            # _, last_h = state
            last_h = state

            # use open gates initially when working with stiefel optimization.
            if self._stiefel:
                bias_init = 4.0
            else:
                bias_init = 0.0

            if self._single_gate:
                r, z = self.single_memory_gate(last_h, inputs, 'single_memory_gate',
                                               bias_init=bias_init, dtype=temp_dtype_)
            else:
                r, z = self.double_memory_gate(last_h, inputs, 'double_memory_gate',
                                               bias_init=bias_init, dtype=temp_dtype_)

            with tf.variable_scope("canditate_h"):
                cinWx = complex_matmul(inputs, self._units, 'wx', bias=False,
                                       reuse=self._reuse, dtype=temp_dtype_)
                rhU = complex_matmul(tf.multiply(r, last_h), self._units, 'rhu',
                                     bias=True, unitary=self._stiefel,
                                     reuse=self._reuse, dtype=temp_dtype_)
                tmp = cinWx + rhU

                h_bar = self._activation(tmp)

            new_h = (1 - z)*last_h + z*h_bar

            output = new_h
            # output = tf.concat([tf.real(new_h), tf.imag(new_h)], axis=-1)
            newstate = URNNStateTuple(output, new_h)

            return output, newstate


