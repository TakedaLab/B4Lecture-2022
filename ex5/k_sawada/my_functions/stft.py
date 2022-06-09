import numpy as np


def stft(audio, fs, nperseg=512, noverlap=256):
    """Short-Time Fourier Transform

    Args:
        audio (array_like): input audio array
        fs (float): sampling rate of input audio
        nperseg (int, optional): samples in stft segment. Defaults to 512.
        noverlap (int, optional): samples of stft overlap. Defaults to 256.

    Returns:
        ndarray: stft of audio
        ndarray: time axis
        ndarray: frequency axis
    """

    # zero padding at end of audio
    padding_length = nperseg - len(audio) % (nperseg - noverlap)
    audio = np.concatenate([audio, np.zeros(padding_length)])

    # create window (hanning)
    window = np.hanning(nperseg)

    # stft
    steps = (len(audio) - noverlap) // (nperseg - noverlap)
    Zxx = np.zeros((steps, nperseg), dtype=np.complex128)
    for i in range(steps):
        # apply window
        sample = window * audio[
            (nperseg - noverlap) * i: (nperseg - noverlap) * i + nperseg
        ]
        # fft
        Zxx[i] = np.fft.fft(sample)

    # create time axis
    t = np.linspace(0, len(audio) / fs, Zxx.shape[0])

    # create frequency axis
    f = np.linspace(0, fs, Zxx.shape[1])

    return Zxx.T, t, f


def istft(Zxx, nperseg=512, noverlap=256):
    """Inverse Short-Time Fourier Transform

    Args:
        Zxx (ndarray): spectrogram
        nperseg (int, optional): samples in stft segment. Defaults to 512.
        noverlap (int, optional): samples of stft overlap. Defaults to 256.

    Returns:
        ndarray: reconstructed audio
    """

    # initialize audio array with zeros
    audio = np.zeros((nperseg - noverlap) * (Zxx.shape[1] - 1) + nperseg)

    # istft
    for i, s in enumerate(Zxx.T):
        audio[
            i * (nperseg - noverlap): i * (nperseg - noverlap) + nperseg
        ] += np.fft.ifft(s).real
    return audio
