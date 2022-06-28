# speech enhancement_CVAE1
# train encoder + decoder simultaneously according to noisy speech
# mix noisy training set
# complex128 calculation


import os
import scipy.io as scio
import sys
sys.path.append(os.path.abspath('..'))
import tensorflow.compat.v1 as tf

from runners import *
from collections import OrderedDict
from libs.costs import cmplx_kld, sample_cmplx_Guassian
from complexnn.complex_dense_dtype import ComplexDense
import random
from tensorflow.python import debug as tf_debug
from tools.audio import complex_spec_to_audio
from complexnn.complex_gru_dtype import StiefelGatedRecurrentUnit as SGRU

# using CUDA:1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# configuration
# noisy_speech_path = r''
noisy_speech_path = r''
clean_speech_path = r''
model_dir = r''
exp_dir = r''
logpath = r'log.txt'
dev_logpath = r'dev_log.txt'
model_path = r'/model/SECRVAE.ckpt'
output_path = r'/saved/'


datatype = 'complex'  # complex
if datatype == 'float':
    _dtype = tf.float32
elif datatype == 'complex':
    _dtype = tf.complex128

# parameter setting
is_train = True
N_epoch = 500
batch_size = 200
npatience = 20
N_dim = 200

stiefel = True
from tensorflow.core.protobuf import rewriter_config_pb2

config = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = off

noise_types = ['bbl', 'caf', 'ssn', 'str']
# noise_types = ['bus', 'ped']


# load training data
temp = clean_speech_path + 'train.mat'
tr_target = scio.loadmat(temp)
tr_target = tr_target['feats']
tr_target = tr_target[:, 1:]
t_len = np.shape(tr_target)[0]
M = t_len // batch_size
t_len1 = M * batch_size
tr_target = tr_target[:t_len1, :]



# load development data
temp = clean_speech_path + 'dev.mat'
dev_target = scio.loadmat(temp)
dev_target = dev_target['feats']
dev_target = dev_target[:, 1:]
dev_t_len = np.shape(dev_target)[0]
dev_M = dev_t_len // batch_size
dev_t_len1 = dev_M * batch_size
dev_target = dev_target[:dev_t_len1, :]



class SECRVAE(object):
    def __init__(self, training=False, **kwargs):
        self.training = training
        super(SECRVAE, self).__init__()
        # create data members
        self._model_conf = {"input_shape": [batch_size, N_dim],
                            "input_dtype": _dtype,
                            "target_shape": [batch_size, N_dim],
                            "target_dtype": _dtype,
                            "cmplx_GRU_z_layer1": 512,
                            "cmplx_fc_z_layer2": 512,
                            "cmplx_GRU_dec1": 512,
                            "cmplx_fc_dec2": N_dim,
                            "n_latent": 512,
                            "if_bn": True}
        self._feed_dict = None  # feed dict needed for outputs
        self._outputs = None  # general outputs
        self._global_step = None  # global_step for saver
        self._ops = None
        # build model
        self._build_model()
        # create saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    def _build_encoder(self, inputs, reuse=False):
        # self._model_conf["input_shape"] = inputs.get_shape().as_list()
        with tf.variable_scope("enc", reuse=reuse):
            encoder_cell = SGRU(units=self._model_conf['cmplx_GRU_z_layer1'],
                                stiefel=stiefel, reuse=reuse, dtype=_dtype)
            init_state = encoder_cell.zero_state(batch_size)
            enc1, _ = encoder_cell(inputs, init_state.h)
            print('Enc: enc1 ', enc1)
            z_para = ComplexDense(units=self._model_conf['n_latent'] * 3, name='z_para', dtype=_dtype)(enc1)
            print('Enc: z_para ', z_para)
        z_mu = z_para[:, :self._model_conf['n_latent']]
        z_r = z_para[:, (self._model_conf['n_latent']):(self._model_conf['n_latent'] * 2)]
        z_s = z_para[:, (self._model_conf['n_latent'] * 2):]
        z = sample_cmplx_Guassian(z_mu, z_r, z_s)
        print('Enc: z ', z)
        return [z_mu, z_r, z_s], z

    def _build_decoder(self, z, reuse=False):
        x = []
        with tf.variable_scope("dec", reuse=reuse):
            decoder_cell = SGRU(units=self._model_conf['cmplx_GRU_dec1'],
                                stiefel=stiefel, reuse=reuse, dtype=_dtype)
            init_state = decoder_cell.zero_state(batch_size)
            dec1, _ = decoder_cell(z, init_state.h)
            print('Dec: dec1 ', dec1)
            tempx = ComplexDense(units=self._model_conf['cmplx_fc_dec2'], name='tempx', dtype=_dtype)(dec1)
            print('Dec: tempx ', tempx)
            tempx = tf.expand_dims(tempx, axis=1)
            x.append(tempx)
            x = tf.concat(x, axis=1, name="recon_x")
        x = tf.reshape(x, self._model_conf["input_shape"])  # [None, 2, 20, 201]
        print('Dec: output ', x)
        return x


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
        sampled_x = self._build_decoder(sampled_z)

        with tf.name_scope("costs"):
            with tf.name_scope("kld_z"):
                kld_z = tf.math.reduce_sum(cmplx_kld(*qz_x), axis=1)

            with tf.name_scope("l2_loss"):
                error = tf.math.abs(sampled_x - targets)
                l2_loss = tf.nn.l2_loss(error)

            with tf.name_scope("total_loss"):
                total_loss = tf.reduce_mean(kld_z + l2_loss)

        self._outputs = {
            # model output
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


# model setting
model = SECRVAE(training=True)
sum_names = ["kld_z", "l2_loss", "total_loss"]
sum_vars = [tf.reduce_mean(model.outputs()[name]) for name in sum_names]

dev_sum_names = ["kld_z", "l2_loss",  "total_loss"]
dev_sum_vars = [tf.reduce_mean(model.outputs()[name]) for name in dev_sum_names]

best_dev_lb = np.inf
indx = random.sample(range(M), M)
dev_indx = random.sample(range(dev_M), dev_M)
# train
with tf.Session(config=SESS_CONF).as_default() as sess:
# sess = tf.Session(config=SESS_CONF)
# with tf_debug.LocalCLIDebugWrapperSession(sess) as sess:
    init_step = model.init_or_restore_model(sess, model_dir)
    global_step = int(init_step)
    train_writer = tf.summary.FileWriter("%s/log/train" % exp_dir, sess.graph)
    dev_writer = tf.summary.FileWriter("%s/log/dev" % exp_dir)
    print("start training...")
    for i in range(N_epoch):
        print('Epoch ', i)
        tr_kld_loss, tr_l2_loss, tr_total_loss = 0.0, 0.0, 0.0
        for noise_type in noise_types:
            print('noise type ', noise_type)
            temp = noisy_speech_path + noise_type + '/train.mat'
            tr_input = scio.loadmat(temp)
            tr_input = tr_input['feats']
            tr_input = tr_input[:, 1:]
            tr_input = tr_input[:t_len1, :]
            indx = random.sample(range(M), M)
            for j in range(M):
                tr_input_ = tr_input[(indx[j] * batch_size):((indx[j] + 1) * batch_size), :]
                tr_input_ = np.reshape(tr_input_, (batch_size, N_dim)).astype(np.complex128)

                tr_target_ = tr_target[(indx[j] * batch_size):((indx[j] + 1) * batch_size), :]
                tr_target_ = np.reshape(tr_target_, (batch_size, N_dim)).astype(np.complex128)

                feed_dict = {
                    model.feed_dict()["inputs"]: tr_input_,
                    model.feed_dict()["targets"]: tr_target_,
                    model.feed_dict()['is_train']: 1
                }

                global_step, outputs_, _ = sess.run(
                    [model.global_step(), sum_vars, model.ops()["train_step"]],
                    feed_dict)

                tr_kld_loss += outputs_[0]
                tr_l2_loss += outputs_[1]
                tr_total_loss += outputs_[2]

        tr_kld_loss = tr_kld_loss / (M * 4)
        tr_l2_loss = tr_l2_loss / (M * 4)
        tr_total_loss = tr_total_loss / (M * 4)

        str = "[epoch %.f]: " % i + \
              '\t'.join(["%s %.4f" % p for p in zip(
                  sum_names, [tr_kld_loss, tr_l2_loss, tr_total_loss])]) + '\n'
        print(str)
        with open(logpath, 'a') as f:
            f.write(str)

        dev_kld_loss, dev_l2_loss, dev_total_loss = 0.0, 0.0, 0.0
        for noise_type in noise_types:
            temp = noisy_speech_path + noise_type + '/dev.mat'
            dev_input = scio.loadmat(temp)
            dev_input = dev_input['feats']
            dev_input = dev_input[:, 1:]
            dev_input = dev_input[:t_len1, :]
            dev_indx = random.sample(range(dev_M), dev_M)

            for k in range(dev_M):
                dev_input_ = dev_input[(dev_indx[k] * batch_size):(dev_indx[k] + 1) * batch_size, :]
                dev_input_ = np.reshape(dev_input_, (batch_size, N_dim)).astype(np.complex128)

                dev_target_ = dev_target[(dev_indx[k] * batch_size):(dev_indx[k] + 1) * batch_size, :]
                dev_target_ = np.reshape(dev_target_, (batch_size, N_dim)).astype(np.complex128)

                dev_feed_dict = {
                    model.feed_dict()["inputs"]: dev_input_,
                    model.feed_dict()["targets"]: dev_target_,
                    model.feed_dict()['is_train']: 0
                }

                dev_outputs_ = sess.run(dev_sum_vars, dev_feed_dict)

                if np.isnan(dev_outputs_[2]):
                    print('at %d epoch, noise_type %s, %d segment in development set ' % (i, noise_type, int(dev_indx[k])))
                    print('Nan exist...')
                    exit()

                dev_kld_loss += dev_outputs_[0]
                dev_l2_loss += dev_outputs_[1]
                dev_total_loss += dev_outputs_[2]

        dev_kld_loss = dev_kld_loss / (dev_M * 4)
        dev_l2_loss = dev_l2_loss / (dev_M * 4)
        dev_total_loss = dev_total_loss / (dev_M * 4)

        dev_str = "[dev epoch %.f]: " % i + \
                  '\t'.join(["%s %.4f" % p for p in zip(
                      dev_sum_names, [dev_kld_loss, dev_l2_loss, dev_total_loss])]) + '\n'
        print(dev_str)

        with open(dev_logpath, 'a') as f:
            f.write(dev_str)



        if dev_total_loss < best_dev_lb:
            print('best dev_lb ', dev_total_loss)
            best_epoch, best_dev_lb = i, dev_total_loss
            model.saver.save(sess, model_path)

        if i - best_epoch > npatience:
            print("early stop exit training")
            break

    model.saver.save(sess, model_path)

# test
# load test data
temp = clean_speech_path + 'test.mat'
tt_target = scio.loadmat(temp)
tt_target = tt_target['feats']
tt_target = tt_target[:, 1:]
t_len = np.shape(tt_target)[0]
M = t_len // batch_size
t_len1 = M * batch_size
tt_target = tt_target[:t_len1, :]


test_files = ['test0.mat', 'test3.mat', 'test-3.mat', 'test-6.mat', 'test6.mat']

with tf.Session(config=SESS_CONF).as_default() as sess:
    model.saver.restore(sess, model_path)
    for noise_type in noise_types:
        for test_file in test_files:
            print('*-'*20)
            print('noise type ', noise_type)
            print('test file ', test_file)
            temp = noisy_speech_path + noise_type + '/' +  test_file
            tt_input = scio.loadmat(temp)
            tt_input = tt_input['feats']
            tt_input = tt_input[:, 1:]
            tt_input = tt_input[:t_len1, :]

            save_output = []

            for i in range(M):
                tt_input_ = tt_input[(i * batch_size):((i + 1) * batch_size), :]
                tt_input_ = np.reshape(tt_input_, (batch_size, N_dim))

                tt_target_ = tt_target[(i * batch_size):((i + 1) * batch_size), :]
                tt_target_ = np.reshape(tt_target_, (batch_size, N_dim))

                feed_dict = {
                    model.feed_dict()["inputs"]: tt_input_,
                    model.feed_dict()["targets"]: tt_target_,
                    model.feed_dict()['is_train']: 0,
                }

                outputs_ = sess.run(
                    [model.outputs()['sampled_x']],
                    feed_dict)
                outputs_ = np.asarray(outputs_)

                outputs_ = np.reshape(outputs_, [batch_size, N_dim])
                save_output.append(outputs_)

            save_output = np.array(save_output)
            save_output = np.reshape(save_output, (-1, np.shape(save_output)[-1]))

            spec_save_path = output_path + 'spec/' + noise_type + '/'
            if not os.path.exists(spec_save_path):
                os.makedirs(spec_save_path)
            spec_save_file = spec_save_path + test_file.split('.')[0] + '.mat'

            mmdict = {'output': save_output}
            scio.savemat(spec_save_file, mmdict)

            # save audio file
            audio_save_path = output_path + 'audio/' + noise_type + '/'
            if not os.path.exists(audio_save_path):
                os.makedirs(audio_save_path)
            audiofile = audio_save_path + test_file.split('.')[0] + '.wav'

            temp = np.zeros((np.shape(save_output)[0], 1)).astype(np.complex)
            save_output = np.concatenate([temp, save_output], axis=-1)
            print('save_output shape ', np.shape(save_output))

            complex_spec_to_audio(save_output, audiofile, frame_size_n=400, shift_size_n=160, fft_size=400)
