import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import librosa
import librosa.display

from my_functions import mfcc
from my_functions import stft


def main():
    parser = argparse.ArgumentParser(description='ex5')
    parser.add_argument("-i", "--input", help="input file name")
    parser.add_argument("-o", "--order", help="mel filter bank order", type=int)
    args = parser.parse_args()
    
    # read audio file
    filename = args.input
    sampling_rate = 16000
    wav, _ = librosa.load(filename, sr=sampling_rate, mono=True)
    
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.45)
    
    # plot stft
    Zxx, t, f = stft.stft(wav, sampling_rate)
    ax[0].set_title("spectrogram")
    im = ax[0].imshow(20 * np.log10(np.abs(np.flipud(Zxx[:Zxx.shape[0] // 2]))),
                      cmap=plt.cm.jet,
                      aspect="auto",
                      extent=[t[0], t[-1], f[0], f[len(f) // 2]])
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Frequency [Hz]")
    # colorbar: https://qiita.com/skotaro/items/01d66a8c9902a766a2c0#axes%E3%81%8C%E8%A4%87%E6%95%B0%E3%81%82%E3%82%8B%E5%A0%B4%E5%90%88
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    # mfcc
    order = args.order
    mfcc_, t = mfcc.mfcc(wav, sampling_rate, order)
    ax[1].set_title("mfcc")
    im = ax[1].imshow(np.flipud(mfcc_),
                      cmap=plt.cm.jet,
                      aspect="auto",
                      extent=[t[0], t[-1], 0, order])
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("mel order")
    # colorbar: https://qiita.com/skotaro/items/01d66a8c9902a766a2c0#axes%E3%81%8C%E8%A4%87%E6%95%B0%E3%81%82%E3%82%8B%E5%A0%B4%E5%90%88
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    
    # delta mfcc
    delta_mfcc = mfcc.delta_multiplication(mfcc_)
    ax[2].set_title("delta mfcc")
    im = ax[2].imshow(np.flipud(delta_mfcc),
                      cmap=plt.cm.jet,
                      aspect="auto",
                      extent=[t[0], t[-1], 0, order])
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("mel order")
    # colorbar: https://qiita.com/skotaro/items/01d66a8c9902a766a2c0#axes%E3%81%8C%E8%A4%87%E6%95%B0%E3%81%82%E3%82%8B%E5%A0%B4%E5%90%88
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    
    # delta-delta mfcc
    delta_delta_mfcc = mfcc.delta_multiplication(delta_mfcc)
    ax[3].set_title("delta-delta mfcc")
    im = ax[3].imshow(np.flipud(delta_delta_mfcc),
                      cmap=plt.cm.jet,
                      aspect="auto",
                      extent=[t[0], t[-1], 0, order])
    ax[3].set_xlabel("Time [s]")
    ax[3].set_ylabel("mel order")
    # colorbar: https://qiita.com/skotaro/items/01d66a8c9902a766a2c0#axes%E3%81%8C%E8%A4%87%E6%95%B0%E3%81%82%E3%82%8B%E5%A0%B4%E5%90%88
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    
    plt.savefig("mfcc.png")
    plt.show()
    

if __name__ == "__main__":
    main()
