###############################################################################################################
# This file includes the model 'RVAE' for real-valued VAE, as mentioned in the paper as baseline.
# The encoder and decoder contains 939 units GRU layers except the decoder output layer.
###############################################################################################################


import os
import sys
sys.path.append(os.path.abspath('..'))
import tensorflow.compat.v1 as tf
import numpy as np
from runners import *
from collections import OrderedDict
from libs.costs import cmplx_kld, sample_cmplx_Guassian, sample_real_Gaussian, kld
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.compat.v1.nn.rnn_cell import GRUCell


datatype = 'float'  # complex
if datatype == 'float':
    _dtype = tf.float32
elif datatype == 'complex':
    _dtype = tf.complex128

N_dim = 200
stiefel = True
SEQ_LEN = 2

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

class SERVAE(object):
    def __init__(self, training=False, **kwargs):
        self.training = training
        super(SERVAE, self).__init__()
        # create data members
        self._model_conf = {"input_shape": [None, SEQ_LEN, N_dim],
                            "input_dtype": _dtype,
                            "target_shape": [None, SEQ_LEN, N_dim],
                            "target_dtype": _dtype,
                            "GRU_z_layer1": 939,
                            "fc_z_layer2": 939,
                            "GRU_dec1": 939,
                            "fc_dec2": N_dim,
                            "n_latent": 939,
                            "if_bn": True}
        self._feed_dict = None  # feed dict needed for outputs
        self._outputs = None  # general outputs (acc, posterior...)
        self._global_step = None  # global_step for saver
        self._ops = None  # accessible ops (train_step, decay_op...)
        # build model
        self._build_model()
        # create saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # encoder for sequence variable - speaker id
    def _build_encoder(self, inputs, reuse=False):
        print('inputs shape ', inputs.get_shape().as_list())
        input_shape = tuple(array_ops.shape(input_) \
                            for input_ in nest.flatten(inputs))
        batch_size = input_shape[0][0]
        num_units = [self._model_conf['GRU_z_layer1'], 939*2]
        z_mu, z_r, z_s, z = [], [], [], []
        with tf.variable_scope("enc", reuse=reuse):
            encoder_cells = [GRUCell(num_units=unit,
                                reuse=reuse, dtype=_dtype) for unit in num_units]
            enc_multi_cells = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)
            enc_init_state = enc_multi_cells.zero_state(batch_size, dtype=_dtype)
            enc1, enc_final_states = tf.nn.dynamic_rnn(
                enc_multi_cells,
                inputs,
                dtype=self._model_conf["input_dtype"],
                initial_state=enc_init_state,
                time_major=False,
                scope="encoder_dynamic_RNN")

            temp_z_para = enc1

            z_mu = temp_z_para[:, :, :self._model_conf['n_latent']]
            z_r = temp_z_para[:, :, (self._model_conf['n_latent']):]
            z = sample_real_Gaussian(z_mu, z_r)
        return [z_mu, z_r], z

    def _build_decoder(self, z, reuse=False):
        input_shape = tuple(array_ops.shape(input_) \
                            for input_ in nest.flatten(z))

        batch_size = input_shape[0][0]
        num_units = [self._model_conf['GRU_z_layer1'], 200]
        with tf.variable_scope("dec", reuse=reuse):
            decoder_cells = [GRUCell(num_units=unit,
                                reuse=reuse, dtype=_dtype, activation='linear') for unit in num_units]
            dec_multi_cells = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)
            dec_init_state = dec_multi_cells.zero_state(batch_size, dtype=_dtype)
            dec1, dec_final_states = tf.nn.dynamic_rnn(
                dec_multi_cells,
                z,
                dtype=self._model_conf["input_dtype"],
                initial_state=dec_init_state,
                time_major=False,
                scope="encoder_dynamic_RNN")

            mean_x = tf.reduce_mean(tf.math.abs(dec1))
        return dec1, mean_x



    def _build_model(self):
        inputs = tf.placeholder(
            self._model_conf["input_dtype"],
            shape=self._model_conf["input_shape"],
            name="inputs")
        targets = tf.placeholder(
            self._model_conf["target_dtype"],
            shape=self._model_conf["target_shape"],
            name="targets")
        is_train = tf.placeholder(tf.bool, name="is_train")

        self._feed_dict = {"inputs": inputs,
                           "targets": targets,
                           "is_train": is_train}

        qz_x, sampled_z = self._build_encoder(inputs)
        sampled_x, mean_x = self._build_decoder(sampled_z)

        with tf.name_scope("costs"):
            # labeled data costs
            with tf.name_scope("kld_z"):
                kld_z = tf.math.reduce_mean(kld(*qz_x), axis=(1, 2))

            with tf.name_scope("l2_loss"):
                error = tf.math.abs(sampled_x - targets)
                l2_loss = tf.nn.l2_loss(error)

            with tf.name_scope("total_loss"):
                total_loss = tf.reduce_mean(kld_z + l2_loss)

        self._outputs = {
            "mean_x": mean_x,
            # model params
            "qz_x": qz_x,
            "sampled_z": sampled_z,
            "sampled_x": sampled_x,
            # costs
            "kld_z": kld_z,
            "l2_loss": l2_loss,
            "total_loss": total_loss
        }
        # create ops for training
        self._build_train()

    def _build_train(self):
        # create grads and clip optionally
        params = tf.trainable_variables()
        self._global_step = tf.get_variable("global_step", trainable=True,
                                            initializer=0.0)
        with tf.name_scope("grad"):
            grads = tf.gradients(self._outputs["total_loss"], params)
            clipped_grads = grads
        self._grads = OrderedDict(
            zip(["grad_%s" % param.name for param in params], clipped_grads))


        # create ops
        with tf.name_scope("train"):
            lr = tf.Variable(0.00001, trainable=False)
            opt_opts = {}
            opt_opts["beta1"] = 0.9
            opt_opts["beta2"] = 0.999
            opt = tf.train.AdamOptimizer(learning_rate=lr, **opt_opts)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = opt.apply_gradients(zip(clipped_grads, params),
                                                 global_step=self._global_step)
            lr = lr.assign(lr * 0.8)
            decay_op = lr

        self._ops = {"train_step": train_step, "decay_op": decay_op}

    def init_or_restore_model(self, sess, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model params from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh params")
            sess.run(tf.global_variables_initializer())
        return sess.run(self._global_step)

    def feed_dict(self):
        return self._feed_dict

    def outputs(self):
        return self._outputs

    def grads(self):
        return self._grads

    def global_step(self):
        return self._global_step

    def ops(self):
        return self._ops

