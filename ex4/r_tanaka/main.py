import numpy as np
import os
import librosa
import librosa.display
import scipy
from scipy import fftpack
import matplotlib.pyplot as plt


def AutoCorrelation(y):
    """autocorrelation

    Args:
        y (ndarray): input data.

    Returns:
        ndarray: result of calculate autocorrelation
    """
    # number of data samples
    ynum = y.shape[0]
    # autocorrelation
    AC = []
    for i in range(ynum):
        if i == 0:
            AC.append(np.sum(y * y))
        else:
            AC.append(np.sum(y[0: -i] * y[i:]))
    return np.array(AC)


def detect_peak(x):
    """detect peak candidates

    Args:
        x (ndarray): input data.

    Returns:
        ndarray: peak candidates
    """
    peak = []
    # exclude first and last from peak
    for i in range(x.shape[0]-2):
        if x[i] < x[i+1] and x[i+1] > x[i+2]:
            peak.append(x[i+1])
    return np.array(peak)


def f0_AutoCorrelation(x, sr, hop_length=512, win_length=1024):
    """find f0 by calculating autocorrelation

    Args:
        x (ndarray): input data
        sr (int): sampling rate
        hop_length (int, optional): hop length. Defaults to 512.
        win_length (int, optional): window size. Defaults to 1024.

    Returns:
        ndarray: f0
    """
    # number of data samples
    xnum = x.shape[0]

    f0 = []
    for i in range(int((xnum - hop_length) / hop_length)):

        # extract the part of array x
        tmp = x[i * hop_length: i * hop_length + win_length]
        # calculate autocorrelation
        AC = AutoCorrelation(tmp)
        # detect peak candidates
        peak = detect_peak(AC)
        # get the index of the largest value
        fundamental_period = np.where(AC == np.max(peak))[0]
        # get f0 data
        f0.append(sr/fundamental_period[0])

    # adjust the length of f0 data
    f0 = np.concatenate([[f0[0]], f0, [f0[-1]]])
    return np.array(f0)


def get_cepstrum(x):
    """get cepstrum

    Args:
        x (ndarray): input data

    Returns:
        ndarray: cepstrum data (db scale)
    """
    spec = fftpack.fft(x)
    spec_db = 20 * np.log10(np.abs(spec))
    ceps_db = fftpack.ifft(spec_db)
    return ceps_db


def f0_cepstrum(x, sr, index=10, hop_length=512, win_length=1024):
    """find f0 by cepstrum analysis

    Args:
        x (ndarray): input data
        sr (int): sampling rate
        index (int, optional): low-pass lifter's index. Defaults to 10.
        hop_length (int, optional): hop length. Defaults to 512.
        win_length (int, optional): window size. Defaults to 1024.

    Returns:
        ndarray: f0
    """
    # number of data samples
    xnum = x.shape[0]
    # set a threshold to eliminate errors
    threshold = 0.75

    f0 = []
    fundamental_period = [100]
    # cepstrum analysis by cutting the array x into windows
    for i in range(int((xnum - hop_length) / hop_length)):
        # extract the part of array x
        tmp_x = x[i * hop_length: i * hop_length + win_length]
        tmp_ceps = get_cepstrum(tmp_x)
        # apply high-pass lifter
        tmp_ceps[: index] = 0
        tmp_ceps[-index:] = 0
        peak = detect_peak(tmp_ceps)
        # update fundamental_period (within the threshold level)
        if np.where(tmp_ceps == np.max(peak))[0][0] < win_length * threshold:
            fundamental_period = np.where(tmp_ceps == np.max(peak))[0]
        # get f0 data
        f0.append(sr/fundamental_period[0])

    # adjust the length of f0 data
    f0 = np.concatenate([[f0[0]], f0, [f0[-1]]])
    return np.array(f0)


def envelope_cepstrum(x, index=30):
    """find spectrum envelope by cepstrum analysis

    Args:
        x (ndarray): input data.
        index (int, optional): low-pass lifter's index. Defaults to 30.

    Returns:
        ndarray: spectrum envelope
    """
    ceps_db = get_cepstrum(x)
    # Apply low-pass lifter
    ceps_db[index: -index] = 0
    env_ceps = fftpack.fft(ceps_db)
    return env_ceps


def LevinsonDurbin(r, order):
    """Levinson-Durbin Recursion Algorithm

    Args:
        r (ndarray): autocorrelation
        order (int): lpc order

    Returns:
        a (ndarray): LPC coefficients
        e (ndarray): residual variance
    """
    # LPC coefficients
    a = np.zeros(order + 1)
    # residual variance
    e = np.zeros(order + 1)

    # k=1 case
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + a[1] * r[1]
    lam = - r[1] / r[0]

    # find case k+1 recursively from case k
    for k in range(1, order):
        # update lambda
        lam = 0.0
        lam = - np.sum(a[: k+1] * np.flipud(r[1: k+2])) / e[k]

        # update 'a' by calculating U & V
        U = [1]
        U.extend(a[1: k+1])
        U.append(0)

        V = [0]
        V.extend(np.flipud(a[1: k+1]))
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # update 'e'
        e[k + 1] = (1.0 - lam * lam) * e[k]

    return a, e[-1]


def envelope_LPC(x, order, nfft):
    """find spectrum envelope by LPC analysis

    Args:
        x (ndarray): input data
        order (int): LPC order
        nfft (int): sample size of fft

    Returns:
        ndarray: spectrum envelope by LPC analysis
    """
    # apply hamming function
    hammingWindow = np.hamming(len(x))
    x = x * hammingWindow

    # find LPC coefficients
    order = 8
    r = AutoCorrelation(x)
    a, e = LevinsonDurbin(r, order)

    # LPC los-scale spectrum
    w, h = scipy.signal.freqz(np.sqrt(e), a, nfft, "whole")
    lpcspec = np.abs(h)
    env_lpc = 20 * np.log10(lpcspec)

    return env_lpc


def main():
    # load audio file
    # get current working directory
    dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    audio_path = dir + "aiueo.wav"
    # get waveform data and sample rate
    wav, sr = librosa.load(audio_path, mono=True)

    # set drawing area
    plt.rcParams["figure.figsize"] = (10, 9)
    fig1, ax1 = plt.subplots(2, 1)
    fig1.tight_layout(rect=[0.05, 0, 1, 0.95])
    fig1.subplots_adjust(hspace=0.2)
    fig2, ax2 = plt.subplots(1, 1)
    fig2.tight_layout(rect=[0.05, 0.2, 0.95, 0.8])

    # get original spectrogram
    spectrogram = librosa.stft(wav, win_length=1024, hop_length=512)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # plot original spectrogram on ax[0] & ax[1]
    for i in 0, 1:
        img = librosa.display.specshow(spectrogram_db, y_axis="log", cmap="rainbow", ax=ax1[i])
        ax1[i].set(xlabel='time[s]', ylabel='frequency [Hz]')
        fig1.colorbar(img, aspect=10, pad=0.01, extend="both", ax=ax1[i], format="%+2.f dB")

    # get f0 by calculating AutoCorrelation
    f0_AC = f0_AutoCorrelation(wav, sr)
    # plot f0 waveform　overlaid on the spectrogram
    ax1[0].plot(np.arange(spectrogram_db.shape[1]), f0_AC, color='b', linewidth=3.0)
    ax1[0].set(title="f0 autocorrelation")

    # get f0 from cepstrum
    f0_ceps = f0_cepstrum(wav, sr)
    # plot f0 waveform overlaid on the spectrogram
    ax1[1].plot(np.arange(spectrogram_db.shape[1]), np.abs(f0_ceps), color='b', linewidth=3.0)
    ax1[1].set(title="f0 cepstrum")

    # plot original spectrum (log scale)
    frequency = np.linspace(0, sr, len(wav))
    spec_db = 20 * np.log10(np.abs(fftpack.fft(wav)))
    ax2.set_xlim(0, sr//2)
    ax2.plot(frequency, spec_db)
    ax2.set(xlabel='frequency [Hz]', ylabel='sound pressure level [dB]')

    # plot spectrum envelope obtained by cepstrum analysis
    ceps_db_low = envelope_cepstrum(wav, index=30)
    ax2.plot(frequency, np.real(ceps_db_low), color='lime', linewidth=2.0, label="cepstrum")

    # get envelope obtained by LPC analysis
    order = 100
    nfft = 1024
    env_lpc = envelope_LPC(wav, order=order, nfft=nfft)
    # plot spectrum envelope obtained by LPC analysis
    fscale = np.fft.fftfreq(nfft, d=1.0 / sr)[:int(nfft/2)]
    ax2.plot(fscale, env_lpc[:int(nfft/2)], color="r", linewidth=2.0, label="LPC")
    ax2.set(title="envelope spectrum")
    ax2.legend()

    fig1.savefig("aiueo_f0.png")
    fig2.savefig("aiueo_envelope.png")
    plt.show()


if __name__ == "__main__":
    main()
