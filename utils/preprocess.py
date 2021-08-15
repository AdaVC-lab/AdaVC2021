# encoding: utf-8

import numpy as np
import librosa
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from scipy.signal import butter, get_window, filtfilt
from settings import CONV_PARAMS


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def calc_stft(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    overlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - overlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    f0 = f0.astype(float).copy()
    # index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0


def quantize_f0(x, num_bins=256):
    x = x.astype(float).copy()
    uv = (x <= 0)
    x[uv] = 0.0
    x = np.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins+1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc


def calc_spec_f0(wav_path):
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=CONV_PARAMS['mel_dim']).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    # x, sr = read(wav_path)
    x, sr = librosa.load(wav_path, sr=16000)
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = filtfilt(b, a, x)
    prng = RandomState(123)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

    # compute spectrogram
    stft = calc_stft(wav).T
    mel_stft = np.dot(stft, mel_basis)
    mel_stft = 20 * np.log10(np.maximum(min_level, mel_stft)) - 16
    mel_spectrogram = (mel_stft + 100) / 100

    # mel_spectrogram = np.clip(mel_spectrogram, 0, 1).astype('float32')
    mel_spectrogram = mel_spectrogram.astype('float32')

    # extract f0
    f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, sr, 256, min=50, max=600, otype=2)
    index_nonzero = (f0_rapt != -1e10)

    # 去除speaker embedding 保留pitch、rhythm
    f0_mean, f0_std = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, f0_mean, f0_std).astype('float32')
    f0_norm = quantize_f0(f0_norm)
    return mel_spectrogram, f0_norm


def calc_mel(x, sr=16000):
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=CONV_PARAMS['mel_dim']).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    y = filtfilt(b, a, x)
    prng = RandomState(123)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

    # compute spectrogram
    stft = calc_stft(wav).T
    mel_stft = np.dot(stft, mel_basis)
    mel_stft = 20 * np.log10(np.maximum(min_level, mel_stft)) - 16
    mel_spectrogram = (mel_stft + 100) / 100

    # mel_spectrogram = np.clip(mel_spectrogram, 0, 1).astype('float32')
    mel_spectrogram = mel_spectrogram.astype('float32')

    return mel_spectrogram


def gen_timestamp(mel_spectrogram):
    sr = 16000
    hop_length = 256
    mel_ms = hop_length / sr * 1000

    ms_along_mel = np.array([mel_ms * i for i in range(mel_spectrogram.shape[0])])
    results = []
    for ms_per_mel in ms_along_mel:
        _, ms = divmod(ms_per_mel, 1000)
        minutes, seconds = divmod(_, 60)
        results.append([minutes, seconds, ms])

    return np.array(results)

