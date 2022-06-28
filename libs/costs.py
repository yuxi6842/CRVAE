from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

datatype = 'complex128'
if datatype == 'complex64':
    floattype = tf.float32
    complextype = tf.complex64
elif datatype == 'complex128':
    floattype = tf.float64
    complextype = tf.complex128

def cmplx_kld(mu1, r1, s1, mu2=None, r2=None, s2=None):
    """compute dimension-wise KL-divergence
    -0.5 (1 + logvar - q_logvar - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar))
    q_mu, q_logvar assumed 0 is set to None
    """
    # q(mu1, r1, s1)
    r1 = tf.math.real(r1)
    r1 = tf.math.exp(r1)
    s1 = tf.math.exp(s1)

    # keep s<r
    abs_s1 = tf.math.abs(s1)
    # new_element = tf.zeros_like(s, dtype=complextype)
    temp = r1 * 0.99 / abs_s1
    if floattype == tf.float64:
        temp = tf.complex(temp, np.double(0.))
    else:
        temp = tf.complex(temp, 0.)
    new_element = s1 * temp
    s1 = tf.where(abs_s1 >= tf.abs(r1), new_element, s1)

    # p(mu2, r2, 0)
    if mu2 is None:
        mu2 = tf.zeros_like(mu1, dtype=complextype)

    if r2 is None:
        r2 = tf.zeros_like(mu1, dtype=complextype)

    r2 = tf.math.real(r2)
    r2 = tf.math.exp(r2)

    # kl divergence calculation
    term1 = tf.math.log(tf.pow(r1, 2) - tf.pow(tf.abs(s1), 2))
    term2 = tf.pow(tf.math.abs(mu1), 2) + r1 + tf.pow(tf.math.abs(mu2), 2) - 2*tf.math.real(mu2)*tf.math.real(mu1) \
            - 2*tf.math.imag(mu1)*tf.math.imag(mu2)

    out = - 0.5*term1 - 1 + tf.math.log(tf.math.abs(r2)) + term2 / r2
    return out



def log_cmplx_normal(x):
    out = - tf.math.log(np.pi) -tf.math.pow(tf.math.abs(x), 2)
    return out



# float64-version sample complex Gaussian
def sample_cmplx_Guassian(mu, r, s):
    ''''
    input(complex values)
        mu -> mean
        r -> covariance
        s -> pseudo covariance
    output(complex values)
        h -> [real_h+1j*imag_h]
    '''
    r = tf.math.real(r)
    r = tf.math.exp(r)
    if s is None:
        s = tf.zeros_like(mu, dtype=complextype)
    else:
        s = tf.math.exp(s)

        # keep s<=r
        abs_s = tf.math.abs(s)
        # new_element = tf.zeros_like(s, dtype=complextype)
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
    f3 = tf.math.sqrt((temp1_f3 - temp2_f3) / (2. * (r + re_s)))#, tf.cast(1e-10, tf.float32))

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

