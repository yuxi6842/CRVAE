import os
import time
import scipy
import numpy as np
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





def complex_spec_to_audio(
        complex_spec, name=None, trim=0, fs=16000,
        frame_size_n=400, shift_size_n=160, fft_size=400, win=None):
    assert (np.asarray(complex_spec).ndim == 2)
    frame_starts_n = np.arange(len(complex_spec)) * shift_size_n
    signal = istft_index(complex_spec, frame_size_n, frame_starts_n, fft_size, win, win)
    if trim > 0:
        signal = signal[trim:-trim]

    if name is not None:
        if os.path.splitext(name)[1] != ".wav":
            name = name + ".wav"
        wavfile.write(name, fs, (signal*32767 / np.max(signal)).astype(np.int16))

    return signal







def check_and_makedirs(dir_path):
    if bool(dir_path) and not os.path.exists(dir_path):
        print("creating directory %s" % dir_path)
        os.makedirs(dir_path)





