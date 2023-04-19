from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

datatype = 'complex128'
if datatype == 'complex64':
    floattype = tf.float32
    complextype = tf.complex64
elif datatype == 'complex128':
    floattype = tf.float64
    complextype = tf.complex128



def cmplx_kld(mu, r, s):
    """compute dimension-wise KL-divergence
    -0.5 (1 + logvar - q_logvar - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar))
    q_mu, q_logvar assumed 0 is set to None
    """
    r = tf.math.real(r)
    r = tf.math.exp(r)
    s = tf.math.exp(s)

    # keep s<r
    abs_s = tf.math.abs(s)
    new_element = tf.zeros_like(s, dtype=complextype)
    s = tf.where(abs_s >= tf.abs(r), new_element, s)

    # kl divergence calculation
    term1 = tf.math.log(tf.pow(r, 2) - tf.pow(tf.abs(s), 2))

    out = - 0.5*term1 - 1 + r + tf.math.pow(tf.math.abs(mu), 2)
    return out

def log_cmplx_normal(x):
    out = - tf.math.log(np.pi) -tf.math.pow(tf.math.abs(x), 2)
    return out


def kld(mu, logvar, q_mu=None, q_logvar=None):
    """compute reaL-valued dimension-wise KL-divergence
    -0.5 (1 + logvar - q_logvar - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar))
    q_mu, q_logvar assumed 0 is set to None
    """
    if q_mu is None:
        q_mu = tf.zeros_like(mu)
    else:
        print("using non-default q_mu %s" % q_mu)

    if q_logvar is None:
        q_logvar = tf.zeros_like(logvar)
    else:
        print("using non-default q_logvar %s" % q_logvar)

    if isinstance(mu, tf.Tensor):
        mu_shape = mu.get_shape().as_list()
    else:
        mu_shape = list(np.asarray(mu).shape)

    if isinstance(q_mu, tf.Tensor):
        q_mu_shape = q_mu.get_shape().as_list()
    else:
        q_mu_shape = list(np.asarray(q_mu).shape)

    if isinstance(logvar, tf.Tensor):
        logvar_shape = logvar.get_shape().as_list()
    else:
        logvar_shape = list(np.asarray(logvar).shape)

    if isinstance(q_logvar, tf.Tensor):
        q_logvar_shape = q_logvar.get_shape().as_list()
    else:
        q_logvar_shape = list(np.asarray(q_logvar).shape)

    if not np.all(mu_shape == logvar_shape):
        raise ValueError("mu_shape (%s) and logvar_shape (%s) does not match" % (
            mu_shape, logvar_shape))
    if not np.all(mu_shape == q_mu_shape):
        raise ValueError("mu_shape (%s) and q_mu_shape (%s) does not match" % (
            mu_shape, q_mu_shape))
    if not np.all(mu_shape == q_logvar_shape):
        raise ValueError("mu_shape (%s) and q_logvar_shape (%s) does not match" % (
            mu_shape, q_logvar_shape))

    return -0.5 * (1 + logvar - q_logvar - \
                   (tf.pow(mu - q_mu, 2) + tf.exp(logvar)) / tf.exp(q_logvar))




def sample_cmplx_Guassian(mu, r, s):
    ''''
    input(complex values)
        mu -> mean
        r -> covariance
        s -> pseudo covariance
    output(complex values)
        h -> [real_h, imag_h]
    '''
    r = tf.math.real(r)
    r = tf.math.exp(r)
    if s is None:
        s = tf.zeros_like(mu, dtype=complextype)
    else:
        s = tf.math.exp(s)

        # keep s<=r
        abs_s = tf.math.abs(s)
        temp = r * 0.99 / abs_s
        if floattype == tf.float64:
            temp = tf.complex(temp, np.double(0.))
        else:
            temp = tf.complex(temp, 0.)
        new_element = s * temp
        s = tf.where(abs_s >= tf.abs(r), new_element, s)

    re_s = tf.math.real(s)
    im_s = tf.math.imag(s)


    xepsilon = tf.random.normal(tf.shape(mu), name='xepsilon', dtype=floattype)
    yepsilon = tf.random.normal(tf.shape(mu), name='yepsilon', dtype=floattype)

    f1 = tf.math.sqrt((r + re_s) / 2.)
    f2 = im_s / (r + re_s)
    temp1_f3 = tf.math.pow(r, 2)
    temp2_f3 = tf.math.pow(tf.math.abs(s), 2)
    f3 = tf.math.sqrt((temp1_f3 - temp2_f3) / (2. * (r + re_s)))

    re_mu = tf.math.real(mu)
    im_mu = tf.math.imag(mu)

    re_h = re_mu + f1 * xepsilon
    im_h = im_mu + f1 * xepsilon * f2 + f3 * yepsilon

    h = tf.complex(re_h, im_h)

    return h

def sample_real_Gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name='epsilon')
    sample = mu + tf.exp(0.5 * logvar) * epsilon
    return sample

