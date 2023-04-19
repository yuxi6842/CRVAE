###############################################################################################################
# This file includes the model 'CVAE' for complex VAE, as mentioned in the paper as baseline.
# The encoder and decoder contains 512 units dense layers except the decoder output layer.
###############################################################################################################

import os
import sys
sys.path.append(os.path.abspath('..'))
import tensorflow.compat.v1 as tf
from collections import OrderedDict
from libs.costs import cmplx_kld, sample_cmplx_Guassian, sample_real_Gaussian, kld

from complexnn.complex_dense_dtype import ComplexDense
from complexnn.activation_functions import mod_relu


datatype = 'complex'  # complex
if datatype == 'float':
    _dtype = tf.float32
elif datatype == 'complex':
    _dtype = tf.complex128


N_dim = 200
stiefel = True
SEQ_LEN = 1

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

class SECRVAE(object):
    def __init__(self, training=False, **kwargs):
        self.training = training
        super(SECRVAE, self).__init__()
        # create data members
        self._model_conf = {"input_shape": [None, N_dim],
                            "input_dtype": _dtype,
                            "target_shape": [None, N_dim],
                            "target_dtype": _dtype,
                            "cmplx_GRU_z_layer1": 512,
                            "cmplx_fc_z_layer2": 512,
                            "cmplx_GRU_dec1": 512,
                            "cmplx_fc_dec2": N_dim,
                            "n_latent": 512,
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
        with tf.variable_scope("enc", reuse=reuse):
            enc1 = ComplexDense(units=self._model_conf['cmplx_GRU_z_layer1'], name='enc_layer1', dtype=_dtype) \
                (inputs)
            Relu_enc1 = mod_relu(enc1)
            enc2 = ComplexDense(units=self._model_conf['n_latent']*3, name='enc_layer2', dtype=_dtype) \
                (Relu_enc1)
            temp_z_para = enc2
            z_mu = temp_z_para[:, :self._model_conf['n_latent']]
            z_r = temp_z_para[:, (self._model_conf['n_latent']):(self._model_conf['n_latent'] * 2)]
            z_s = temp_z_para[:, (self._model_conf['n_latent'] * 2):]
            z = sample_cmplx_Guassian(z_mu, z_r, z_s)

        return [z_mu, z_r, z_s], z

    def _build_decoder(self, z, reuse=False):
        with tf.variable_scope("dec", reuse=reuse):
            dec1 = ComplexDense(units=self._model_conf['cmplx_GRU_z_layer1'], name='dec_layer1', dtype=_dtype) \
                (z)
            Relu_dec1 = mod_relu(dec1)
            dec2 = ComplexDense(units=N_dim, name='dec_layer2', dtype=_dtype) \
                (Relu_dec1)

            mean_x = tf.reduce_mean(tf.math.abs(dec1))
        return dec2, mean_x



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
                kld_z = tf.math.reduce_mean(cmplx_kld(*qz_x), axis=(1))

            with tf.name_scope("RI_loss"):
                Rerror = tf.math.reduce_sum(tf.math.abs(tf.math.real(sampled_x) - tf.math.real(targets)))
                Ierror = tf.math.reduce_sum(tf.math.abs(tf.math.imag(sampled_x) - tf.math.imag(targets)))
                RI_loss = (Rerror + Ierror) / (1*200)

            with tf.name_scope("mag_loss"):
                error = tf.math.abs(tf.abs(sampled_x) - tf.abs(targets))
                mag_loss = tf.math.reduce_sum(error)/ (1*200)

            with tf.name_scope("l2_loss"):
                error = tf.math.abs(sampled_x - targets)
                l2_loss = tf.nn.l2_loss(error)/ (1*200)


            with tf.name_scope("total_loss"):
                total_loss = tf.reduce_mean(kld_z + RI_loss + mag_loss)

        self._outputs = {
            "mean_x": mean_x,
            # model params
            "qz_x": qz_x,
            "sampled_z": sampled_z,
            "sampled_x": sampled_x,
            # costs
            "kld_z": kld_z,
            "RI_loss": RI_loss,
            "mag_loss": mag_loss,
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

