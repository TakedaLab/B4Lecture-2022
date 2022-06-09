import numpy as np
from scipy.fftpack import dct

from my_functions import stft


def f2melf(f):
    """convert frequency into mel frequency

    Args:
        f (int/float): frequency

    Returns:
        float: mel frequency
    """
    return 1000 * np.log(1 + f / 1000) / np.log(2.0)


def melf2f(melf):
    """convert mel frequency into frequency

    Args:
        melf (int/float): mel frequency

    Returns:
        float: frequency
    """
    return np.exp(melf * np.log(2) / 1000) * 1000 - 1000


def mfcc(wav, fs, order, nperseg=512, noverlap=256):
    """mfcc

    Args:
        wav (ndarray, axis=(time)): input data
        fs (int): sampling rate
        order (int): order in mel filter bank
        nperseg (int, optional): samples in stft segment. Defaults to 512.
        noverlap (int, optional): samples in stft overlap. Defaults to 256.

    Returns:
        ndarray, axis=(time, frequency): mfcc
        ndarray, axis=(time, ): time
    """
    # log power spectrum
    spec, t, f = stft.stft(wav, fs, nperseg, noverlap)
    spec = np.power(np.abs(spec), 2)
    
    # get filter bank
    fb_half = create_mel_filterbank(order, fs, nperseg // 2)
    
    # multiply filter bank
    spec_fb = np.empty((0, order))
    for i in np.log10(spec).T:
        spec_fb = np.append(spec_fb, [np.dot(i[:nperseg // 2], fb_half.T)],
                            axis=0)

    # discrete cosine transform
    out = np.empty((0, order))
    for i in spec_fb:
        out = np.append(out, [dct(i)], axis=0)
    return out.T, t


def create_mel_filterbank(order, fs, nperseg=512):
    """mel filter bank

    Args:
        order (int): order in mel filter bank
        fs (int): sampling rate
        nperseg (int, optional): samples in stft segment. Defaults to 512.

    Returns:
        ndarray, axis=(order, frequency): mel filter bank
    """
    # create empty filter bank
    fb = np.zeros((order, nperseg))
    # convert nyquist frequency to mel frequency
    melf_max = f2melf(fs / 2)
    # filterbank length at mel frequency scale
    melf_step = melf_max // (order + 1)
    # get mel frequency index between poek and bottom in filter bank
    critical_melf_point = np.empty(0)
    for i in range(order + 2):
        critical_melf_point = np.append(critical_melf_point, melf_step * i)

    # convert index to frequency
    critical_f_point = (melf2f(critical_melf_point) * nperseg * 2 / fs) \
        .astype("int64")
        
    # create filter bank
    for i in range(order):
        fb[i][int(critical_f_point[i]):int(critical_f_point[i + 1])] += \
            np.linspace(0, 1,
                        (int(critical_f_point[i + 1]) -
                         int(critical_f_point[i])))
        fb[i][int(critical_f_point[i + 1]):int(critical_f_point[i + 2])] += \
            np.linspace(1, 0,
                        (int(critical_f_point[i + 2]) -
                         int(critical_f_point[i + 1])))
    return fb


def delta_multiplication(mfcc, d_start=-2, d_stop=3, d_step=5):
    """delta (dynamic fluctuation component)

    Args:
        mfcc (ndarray, axis=(time, mel frequency)): mfcc
        d_start (int, optional): first value of multiplier array. Defaults to -2.
        d_stop (int, optional): last value of multiplier array. Defaults to 3.
        d_step (int, optional): length of multiplier array. Defaults to 5.

    Returns:
        ndarray, axis=(time, mel frequency)): delta of input
    """
    rows = np.shape(mfcc)[0]
    columns = np.shape(mfcc)[1]
    out = np.zeros((rows, columns - d_step + 1))
    multiplier = np.linspace(d_start, d_stop, d_step)
    for i in range(rows):
        for j in range(columns - d_step):
            out[i, j + 1] = np.sum(mfcc[i][j: j + d_step] * multiplier)
    return out
