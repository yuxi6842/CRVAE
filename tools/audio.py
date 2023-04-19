import os
import time
import scipy
import numpy as np
from scipy import signal
from scipy.io import wavfile




def mel_scale(freq):
    return 1127.0 * np.log(1.0 + float(freq) / 700)


def inv_mel_scale(mel_freq):
    return 700 * (np.exp(float(mel_freq) / 1127) - 1)


def hann(n):
    """
    n   : length of the window
    """
    w = np.zeros(n)
    for x in range(n):
        w[x] = 0.5 * (1 - np.cos(2 * np.pi * x / n))
    return w


def stft_index(wave, frame_size_n, frame_starts_n, fft_size=None, win=None):
    """
    wave            : 1-d float array
    frame_size_n    : number of samples in each frame
    frame_starts_n  : a list of int denoting starting sample index of each frame
    fft_size        : number of frequency bins
    win             : windowing function on amplitude; len(win) == frame_size_n
    """
    wave = np.asarray(wave)
    frame_starts_n = np.int32(frame_starts_n)
    if fft_size is None:
        fft_size = frame_size_n
    if win is None:
        win = np.sqrt(hann(frame_size_n))
    # sanity check
    if not wave.ndim == 1:
        raise ValueError('wave is not mono')
    elif not frame_starts_n.ndim == 1:
        raise ValueError('frame_starts_n is not 1-d')
    elif not len(win) == frame_size_n:
        raise ValueError('win does not match frame_starts_n (%s != %s)', len(win), frame_size_n)
    elif fft_size % 2 == 1:
        raise ValueError('odd ffts not yet implemented')
    elif np.min(frame_starts_n) < 0 or np.max(frame_starts_n) > wave.shape[0] - frame_size_n:
        raise ValueError('Your starting indices contain values outside the allowed range')

    spec = np.asarray([scipy.fft(wave[n:n + frame_size_n] * win, n=fft_size)[:int(fft_size / 2) + 1] \
                       for n in frame_starts_n])
    return spec


def istft_index(spec, frame_size_n, frame_starts_n, fft_size=None, win=None, awin=None):
    """
    spec            : 1-d complex array
    frame_size_n    : number of samples in each frame
    frame_starts_n  : a list of int denoting starting sample index of each frame
    fft_size        : number of frequency bins
    win             : windowing function on spectrogram; len(win) == frame_size_n
    awin            : original windowing function on amplitude; len(win) == frame_size_n
    """
    frame_starts_n = np.int32(frame_starts_n)
    if fft_size is None:
        fft_size = frame_size_n
    if win is None:
        win = np.sqrt(hann(frame_size_n))
    if awin is None:
        awin = np.sqrt(hann(frame_size_n))
    pro_win = win * awin

    # sanity check
    if not frame_starts_n.ndim == 1:
        raise ValueError('frame_starts_n is not 1-d')
    elif not len(win) == frame_size_n:
        raise ValueError('win does not match frame_starts_n (%s != %s)', len(win), frame_size_n)
    elif not len(awin) == frame_size_n:
        raise ValueError('awin does not match frame_starts_n (%s != %s)', len(win), frame_size_n)
    elif spec.shape[0] < frame_starts_n.shape[0]:
        raise ValueError('Number of frames in the spectrogram cannot be \
                          less than the size of frame starts')

    N = frame_starts_n[-1] + frame_size_n

    signal = np.zeros(N)
    normalizer = np.zeros(N, dtype=np.float32)

    n_range = np.arange(frame_size_n)
    for i, n_offset in enumerate(frame_starts_n):
        frames = np.real(scipy.ifft(np.concatenate((spec[i], spec[i][-2:0:-1].conjugate())),
                                    n=fft_size))
        signal[n_offset + n_range] += frames * win
        normalizer[n_offset + n_range] += pro_win

    nonzero = np.where(normalizer > 0)
    rest = np.where(normalizer <= 0)
    signal[nonzero] = signal[nonzero] / normalizer[nonzero]
    signal[rest] = 0
    return signal


def comp_spec_image(wave, decom, frame_size_n, shift_size_n, fft_size, awin, log_floor):
    """
    RETURN:
        float matrix of shape (2, T, F)
    """
    frame_starts_n = np.arange(0, wave.shape[0] - frame_size_n, step=shift_size_n)
    wave = wave / np.max(np.abs(wave))
    spec = stft_index(wave, frame_size_n, frame_starts_n, fft_size, awin)
    if decom == "mp":
        phase = np.angle(spec)
        dbmag = np.log10(np.absolute(spec))
        # print("max amplitude %s, max magnitude %s, max phase %s" % (
        #     np.max(wave), np.max(np.absolute(spec)), np.max(phase)))
        dbmag[dbmag < log_floor] = log_floor
        dbmag = 20 * dbmag
        spec_image = np.concatenate([dbmag[None, ...], phase[None, ...]], axis=0)
    elif decom == "ri":
        real = np.real(spec)
        imag = np.imag(spec)
        spec_image = np.concatenate([real[None, ...], imag[None, ...]], axis=0)
    #         print("max amplitude %s, max real %s, max imag %s" % (
    #             np.max(wave), np.max(np.absolute(real)), np.max(np.absolute(imag))))
    else:
        raise ValueError("decomposition type %s not supported" % decom)
    return spec_image


def est_phase_from_mag_spec(
        mag_spec, frame_size_n, shift_size_n, fft_size,
        awin=None, k=1000, min_avg_diff=1e-9, debug=False):
    """
    for quality min_avg_diff 1e-9 is recommended

    mag_spec    - magnitude spectrogram (in linear) of shape (n_time, n_frequency)
    """
    start_time = time.time()
    debug_x = []
    frame_starts_n = np.arange(len(mag_spec)) * shift_size_n
    X_phase = None
    X = mag_spec * np.exp(1j * np.random.uniform(-np.pi, np.pi, mag_spec.shape))
    x = istft_index(X, frame_size_n, frame_starts_n, fft_size, awin, awin)
    for i in range(k):
        X_phase = np.angle(stft_index(x, frame_size_n, frame_starts_n, fft_size, awin))
        X = mag_spec * np.exp(1j * X_phase)
        new_x = istft_index(X, frame_size_n, frame_starts_n, fft_size, awin, awin)
        avg_diff = np.mean((x - new_x) ** 2)
        x = new_x

        if avg_diff < min_avg_diff:
            break

        if debug and i % 100 == 0:
            print("done %s iterations, avg_diff is %s" % (i, avg_diff))
            debug_x.append(x)
    if debug:
        print("time elapsed = %.2f" % (time.time() - start_time))

    return X_phase, debug_x


def convert_to_complex_spec(
        X, X_phase, decom, phase_type, add_dc=False, est_phase_opts=None):
    """
    X/X_phase       - matrix of shape (..., n_channel, n_time, n_frequency)
    decom           - `mp`: magnitude (in dB) / phase (in rad) decomposition
                      `ri`: real / imaginary decomposition
    phase_type      - `true`: X's n_channel = 2
                      `oracle`: use oracle phase X_phase
                      `zero`: use zero matrix as the phase matrix for X
                      `rand`: use random matrix as the phase matrix for X
                      `est`: estimate the phase from magnitude spectrogram
    est_phase_opts  - arguments for est_phase_from_mag_spec

    complex_X is [..., t, f]
    """
    X, X_phase = np.asarray(X), np.asarray(X_phase)

    if X.shape[-3] != 1 and X.shape[-3] != 2:
        raise ValueError("X's n_channel must be 1 or 2 (%s)" % X.shape[-3])
    if np.any(np.iscomplex(X)):
        raise ValueError("X should not be complex")
    if np.any(np.iscomplex(X_phase)):
        raise ValueError("X_phase should not be complex")

    if add_dc:
        X_dc = np.zeros(X.shape[:-1] + (1,))
        X = np.concatenate([X_dc, X], axis=-1)
        if X_phase:
            X_phase_dc = np.zeros(X_phase.shape[:-1] + (1,))
            X_phase = np.concatenate([X_phase_dc, X_phase], axis=-1)

    if decom == "mp":
        print('now in mp mode')
        X_lin_mag = 10 ** (X[..., 0, :, :] / 20)
        if phase_type == "true" and X.shape[-3] != 2:
            raise ValueError("X should have 2 channels for phase_type %s" % (
                phase_type,) + " (X shape is %s)" % (X.shape,))
            X_phase = X[..., 1, :, :]
        else:
            if X.shape[-3] != 1:
                print("WARNING: ignoring X's second channel (phase)")

            if phase_type == "oracle":
                if X_phase is None:
                    raise ValueError("X_phase shape %s invalid for phase_type %s" % (
                        X_phase.shape, phase_type))
            elif phase_type == "zero":
                X_phase = np.zeros_like(X_lin_mag)
            elif phase_type == "rand":
                X_phase = np.random.uniform(-np.pi, np.pi, X_lin_mag.shape)
            elif phase_type == "est":
                X_phase, _ = est_phase_from_mag_spec(X_lin_mag, debug=True, **est_phase_opts)
                print("X_lin_mag shape %s" % (X_lin_mag.shape,))
                print("X_phase shape %s" % (X_phase.shape,))
            else:
                raise ValueError("invalid phase type (%s)" % phase_type)
        complex_X = X_lin_mag * np.exp(1j * X_phase)
    elif decom == "ri":
        print('now the spectrogram is ri model')
        if phase_type != "true":
            raise ValueError("invalid phase type %s. only `true` is valid" % phase_type)
        complex_X = X[..., 0, :, :] + 1j * X[..., 1, :, :]
    else:
        raise ValueError("invalid decomposition %s (mp|ri)" % decom)

    return complex_X

def deemphasis(y, *, coef=0.97, zi=None, return_zf=False):

    b = np.array([1.0, -coef], dtype=y.dtype)
    a = np.array([1.0], dtype=y.dtype)

    if zi is None:
        # initialize with all zeros
        zi = np.zeros(list(y.shape[:-1]) + [1], dtype=y.dtype)
        y_out, zf = signal.lfilter(a, b, y, zi=zi)

        # factor in the linear extrapolation
        y_out -= (
            ((2 - coef) * y[..., 0:1] - y[..., 1:2])
            / (3 - coef)
            * (coef ** np.arange(y.shape[-1]))
        )

    else:
        zi = np.atleast_1d(zi)
        y_out, zf = signal.lfilter(a, b, y, zi=zi.astype(y.dtype))

    if return_zf:
        return y_out, zf
    else:
        return y_out


def complex_spec_to_audio(
        complex_spec, name=None, trim=0, fs=16000,
        frame_size_n=400, shift_size_n=160, fft_size=400, win=None, alpha=0.8):
    assert (np.asarray(complex_spec).ndim == 2)
    frame_starts_n = np.arange(len(complex_spec)) * shift_size_n
    signal = istft_index(complex_spec, frame_size_n, frame_starts_n, fft_size, win, win)
    signal = deemphasis(signal, coef=alpha)
    if trim > 0:
        signal = signal[trim:-trim]

    if name is not None:
        if os.path.splitext(name)[1] != ".wav":
            name = name + ".wav"
        wavfile.write(name, fs, (signal*32767 / np.max(signal)).astype(np.int16))

    return signal


def flatten_channel(utt_feats):
    """
    convert a 3D tensor of (C, T, F) to (T, F_c1+F_c2+...)
    """
    assert (isinstance(utt_feats, np.ndarray) and utt_feats.ndim == 3)
    return np.concatenate(utt_feats, axis=1)


def unflatten_channel(utt_feats, n_chan):
    """
    convert a 2D flattened tensor of (T, F_c1+F_c2+...) to (C, T, F)
    """
    assert (isinstance(utt_feats, np.ndarray) and utt_feats.ndim == 2)

    # if n_chan=2, the return 'utt_feats' has two parts, real and imag
    utt_feats = utt_feats.reshape((utt_feats.shape[0], n_chan, -1))
    utt_feats = utt_feats.transpose((1, 0, 2))
    return utt_feats





def check_and_makedirs(dir_path):
    if bool(dir_path) and not os.path.exists(dir_path):
        print("creating directory %s" % dir_path)
        os.makedirs(dir_path)





