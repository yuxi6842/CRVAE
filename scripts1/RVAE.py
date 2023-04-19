########################################################################################################################
# File to reproduce the baseline 'RVAE' as mentioned in the paper.
########################################################################################################################

import os
import scipy.io as scio
import sys
sys.path.append(os.path.abspath('..'))
import tensorflow.compat.v1 as tf
import numpy as np
from runners import *
from tools.audio import convert_to_complex_spec, complex_spec_to_audio
from models1.CLAAUDIA_home_SE_model9 import SERVAE
import pickle


noisy_speech_path = # datapath, includes clean (['Cfeats']) and noisy (['Nfeats']) spectrogram
model_dir = './model/' # path to save models
exp_dir = './exp_dir/'
logpath = './log.txt'
dev_logpath = './dev_log.txt'
model_path = './model/SECRVAE.ckpt'
output_path = './saved/'


for folder in [model_dir, exp_dir, output_path]:
    if not os.path.exists(folder):
        os.makedirs(folder)



datatype = 'float'  # complex
if datatype == 'float':
    _dtype = tf.float32
elif datatype == 'complex':
    _dtype = tf.complex128

# parameter setting
is_train = True
N_epoch = 500
batch_size = 50  # 4096 5120
npatience = 50
N_dim = 200
SEQ_LEN = 2

stiefel = True


config = tf.ConfigProto()
from tensorflow.core.protobuf import rewriter_config_pb2
config = tf.ConfigProto()

off = rewriter_config_pb2.RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = off

noise_types = ['bbl', 'caf', 'ssn', 'str']


# model setting
model = SERVAE(training=True)
sum_names = ["kld_z", "l2_loss", "total_loss"]
sum_vars = [tf.reduce_mean(model.outputs()[name]) for name in sum_names]

dev_sum_names = ["kld_z", "l2_loss", "total_loss"]
dev_sum_vars = [tf.reduce_mean(model.outputs()[name]) for name in dev_sum_names]

best_dev_lb = np.inf

# train
with tf.Session(config=config).as_default() as sess:
# sess = tf.Session(config=SESS_CONF)
# with tf_debug.LocalCLIDebugWrapperSession(sess) as sess:
    init_step = model.init_or_restore_model(sess, model_dir)
    global_step = int(init_step)
    train_writer = tf.summary.FileWriter("%s/log/train" % exp_dir, sess.graph)
    dev_writer = tf.summary.FileWriter("%s/log/dev" % exp_dir)
    print("start training...")
    for nepoch in range(N_epoch):
        print('Epoch ', nepoch)
        tr_kld_loss, tr_l2_loss, tr_total_loss = 0.0, 0.0, 0.0
        for noise_type in noise_types:
            print('noise type ', noise_type)
            # load dataset
            temp = noisy_speech_path + noise_type + '/train.pkl'
            with open(temp, 'rb') as reader:
                temp_data = pickle.load(reader)

            Ctrfeats = 20*np.log10(np.abs(temp_data['Cfeats'])) # [11240, 100, 200]
            Ntrfeats = 20*np.log10(np.abs(temp_data['Nfeats']))
            del temp_data


            for i in range(0, len(Ctrfeats), batch_size):
                for j in range(0, np.shape(Ctrfeats)[1]-SEQ_LEN, SEQ_LEN):
                    temp_Ntrfeats = Ntrfeats[i:i+batch_size, j:j+SEQ_LEN, :]
                    temp_Ctrfeats = Ctrfeats[i:i+batch_size, j:j+SEQ_LEN, :]
                    feed_dict = {
                        model.feed_dict()["inputs"]: temp_Ntrfeats,
                        model.feed_dict()["targets"]: temp_Ctrfeats,
                        model.feed_dict()['is_train']: 1
                    }
                    global_step, outputs_, newfeats, _ = sess.run(
                        [model.global_step(), sum_vars, model.outputs()['sampled_x'], model.ops()["train_step"]],
                        feed_dict)
                    newfeats = np.asarray(newfeats)
                    tr_kld_loss += outputs_[0]
                    tr_l2_loss += outputs_[1]
                    tr_total_loss += outputs_[2]

        tr_kld_loss = tr_kld_loss / (np.shape(Ctrfeats)[0]//batch_size)
        tr_l2_loss = tr_l2_loss / (np.shape(Ctrfeats)[0] // batch_size)
        tr_total_loss = tr_total_loss / (np.shape(Ctrfeats)[0]//batch_size)

        str = "[epoch %.f]: " % nepoch + \
              '\t'.join(["%s %.4f" % p for p in zip(
                  sum_names, [tr_kld_loss, tr_l2_loss, tr_total_loss])]) + '\n'
        print(str)
        with open(logpath, 'a') as f:
            f.write(str)

        dev_kld_loss, dev_l2_loss, dev_total_loss = 0.0, 0.0, 0.0
        for noise_type in noise_types:
            temp = noisy_speech_path + noise_type + '/dev.pkl'
            with open(temp, 'rb') as reader:
                temp_data = pickle.load(reader)

            Cdevfeats = 20*np.log10(np.abs(temp_data['Cfeats']))
            Ndevfeats = 20*np.log10(np.abs(temp_data['Nfeats']))
            del temp_data


            for i in range(0, len(Cdevfeats), batch_size):
                for j in range(0, np.shape(Cdevfeats)[1]-SEQ_LEN, SEQ_LEN):
                    temp_Ndevdata = Ndevfeats[i:i+batch_size, j:j+SEQ_LEN, :]
                    temp_Cdevdata = Cdevfeats[i:i+batch_size, j:j+SEQ_LEN, :]
                    dev_feed_dict = {
                        model.feed_dict()["inputs"]: temp_Ndevdata,
                        model.feed_dict()["targets"]: temp_Cdevdata,
                        model.feed_dict()['is_train']: 0
                    }
                    dev_outputs_ = sess.run(dev_sum_vars, dev_feed_dict)

                    if np.isnan(dev_outputs_[0]):
                        print('at %d epoch, noise_type %s in development set ' % (nepoch, noise_type))
                        print('Nan exist...')
                        exit()

                    dev_kld_loss += dev_outputs_[0]
                    dev_l2_loss += dev_outputs_[1]
                    dev_total_loss += dev_outputs_[2]

        dev_kld_loss = dev_kld_loss / (np.shape(Cdevfeats)[0]//batch_size)
        dev_l2_loss = dev_l2_loss / (np.shape(Cdevfeats)[0] // batch_size)
        dev_total_loss = dev_total_loss / (np.shape(Cdevfeats)[0]//batch_size)

        dev_str = "[dev epoch %.f]: " % nepoch + \
                  '\t'.join(["%s %.4f" % p for p in zip(
                      dev_sum_names, [dev_kld_loss, dev_l2_loss, dev_total_loss])]) + '\n'
        print(dev_str)

        with open(dev_logpath, 'a') as f:
            f.write(dev_str)

        if dev_total_loss < best_dev_lb:
            print('best dev_lb ', dev_total_loss)
            best_epoch, best_dev_lb = nepoch, dev_total_loss
            model.saver.save(sess, model_path)

        if nepoch - best_epoch > npatience:
            print("early stop exit training")
            break

    model.saver.save(sess, model_path)

# test
temppath = # data to test
noise_types = ['bbl', 'caf', 'ssn', 'str', 'bus', 'ped']
test_files = ['test0.mat', 'test3.mat', 'test-3.mat', 'test-6.mat', 'test6.mat']

###########################################################################################
#                        speech enhancement                                               #
###########################################################################################

with tf.Session(config=SESS_CONF).as_default() as sess:
    model.saver.restore(sess, model_path)
    print('*-'*20)
    for noise_type in noise_types:
        for test_file in test_files:
            tempfile = temppath + noise_type + '/' + test_file
            temp_data = scio.loadmat(tempfile)

            Nttfeats = temp_data['feats'][:, 1:]
            del temp_data

            datalen = len(Nttfeats) // SEQ_LEN
            Cttfeats = 20.0 * np.log10(np.abs(Nttfeats[:5000, :]))
            Cttfeats = np.reshape(Cttfeats, (-1, SEQ_LEN, 200))
            Nttphase = np.angle(Nttfeats[:5000, :])

            save_output = np.empty((0, 200), dtype=np.float32)
            orig_input = np.empty((0, 200), dtype=np.float32)


            for i in range(0, len(Cttfeats)):
                temp_Cttdata = Cttfeats[i]
                temp_Cttdata = np.reshape(temp_Cttdata, (1, -1, N_dim))

                feed_dict = {
                    model.feed_dict()["inputs"]: temp_Cttdata,
                    model.feed_dict()["targets"]: temp_Cttdata,
                    model.feed_dict()['is_train']: 0
                }

                outputs_ = sess.run(
                    [model.outputs()['sampled_x']],
                    feed_dict)

                outputs_ = np.asarray(outputs_)
                outputs_ = np.reshape(outputs_, [-1, N_dim])
                save_output = np.concatenate((save_output, outputs_), axis=0)

            save_output = np.array(save_output)

            save_output = np.reshape(save_output, (-1, np.shape(save_output)[-1]))
            save_output = 10 ** (save_output / 20.0)
            save_output1 = save_output

            # save spectrogram file
            spec_save_path = output_path + 'spec/' + noise_type + '/'
            if not os.path.exists(spec_save_path):
                os.makedirs(spec_save_path)
            spec_save_file = spec_save_path + test_file
            mmdict = {'output': save_output}
            scio.savemat(spec_save_file, mmdict)

            # save audio file
            audio_save_path = output_path + 'audio/' + noise_type + '/'
            if not os.path.exists(audio_save_path):
                os.makedirs(audio_save_path)
            save_output = save_output1 * np.exp(1j * Nttphase)
            audiofile = audio_save_path + test_file.split('.')[0] + '.wav'
            temp = np.zeros((np.shape(save_output)[0], 1)).astype(np.complex)
            save_output1 = np.concatenate([temp, save_output], axis=-1)

            complex_spec_to_audio(save_output1, audiofile, frame_size_n=400, shift_size_n=160, fft_size=400, alpha=0)



