import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import soundfile as sf
import matplotlib.ticker


def STFT(signal,window,step):
    Z = []
    win_fc=np.hamming(window)
    for i in range((signal.shape[0] - window) // step):
        tmp = signal[i*step : i*step + window]
        tmp = tmp * win_fc
        tmp = np.fft.fft(tmp)
        Z.append(tmp)
    Z= np.array(Z)
    return Z


def ISTFT(y,frame_length,window,step):
    Z = np.zeros(frame_length)
    for i in range(len(y)) :
        tmp = np.fft.ifft(y[i])
        Z[i*step : i*step + window] += np.real(tmp)
    Z= np.array(Z)
    return Z


def HPF(fq,sr,fir_size = 512):
    pi = np.pi
    omega = 2 * pi * fq / sr
    arange = np.arange(-fir_size//2,fir_size//2)
    filter = np.sinc(arange) - omega * np.sinc(omega*arange / pi)/pi
    window = np.hamming(fir_size)
    return filter * window


def LPF(fq,sr,fir_size = 512):
    pi = np.pi
    omega = 2 * pi * fq / sr
    arange = np.arange(-fir_size//2,fir_size//2)
    filter = omega * np.sinc(omega*arange / pi)/pi
    window = np.hamming(fir_size)
    return filter * window


def BPF(low,high,sr,fir_size = 512):
    pi = np.pi
    low_omega = 2 * pi * low / sr
    high_omega = 2 * pi * high / sr
    arange = np.arange(-fir_size//2,fir_size//2)
    filter = (high_omega * np.sinc(high_omega*arange / pi) - low_omega * np.sinc(low_omega*arange / pi))/pi
    window = np.hamming(fir_size)
    return filter * window


def BEF(low,high,sr,fir_size = 512):
    pi = np.pi
    low_omega = 2 * pi * low / sr
    high_omega = 2 * pi * high / sr
    arange = np.arange(-fir_size//2,fir_size//2)
    filter = np.sinc(arange) - (high_omega * np.sinc(high_omega*arange / pi) + low_omega * np.sinc(low_omega*arange / pi))/pi
    window = np.hamming(fir_size)
    return filter * window


def convolution(input,filter):
    input_len = len(input)
    filter_len = len(filter)
    result = np.zeros(input_len + filter_len - 1)

    for i in range(input_len):
        result[i : i+filter_len] += np.multiply(input[i] , filter)

    return result


def spectrogram(ax,spec,frame_length,sr):
    spec_log = 20 * np.log10(np.abs(spec).T)[window // 2:] #dB
    im = ax.imshow(spec_log,cmap='jet',extent = [0,frame_length // sr,0,sr // 2,] ,aspect="auto")
    ax.set_yscale("log",base=2)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim([50,20000])
    ax.set_xlabel('Time[s]')
    ax.set_ylabel('Frequency[Hz]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', '2%', pad=0.1)
    cbar = fig.colorbar(im, format='%+2.0f dB', cax=cax)
    cbar.set_label("Magnitude[dB]")
    ax.set_title("Spectrogram")


if __name__ == "__main__":
    #load file
    file_name = "audio.wav"
    window = 1024
    step = window//2

    #original_signal = 音声信号の値、sr=サンプリング周波数 を取得
    original_signal, sr = librosa.load(file_name,sr=None)
    frame_length = original_signal.shape[0]

    #時間軸
    time = np.arange(0,original_signal.shape[0]) / sr

    #STFT
    original_spec = STFT(original_signal,window,step)


    filter =HPF(3000,sr=sr,fir_size = 512)
    filter_property = np.fft.fft(filter,sr) #ここの引数srの有無でいろいろ変わるっぽい？
    filtered_signal = convolution(original_signal,filter) #時間領域
    filtered_spec = STFT(filtered_signal,window,step) #周波数領域に変換

    #PLOT
    fig = plt.figure(figsize=(8,6))

    #Original Signal
    ax1 = fig.add_subplot(3,2,1)
    ax1.plot(time,original_signal)
    ax1.set_xlabel("Time[s]")
    ax1.set_ylabel("Sound Amplitude")
    ax1.set_title("Original Signal")

    #Original Spectrogram
    ax2 = fig.add_subplot(3,2,2)
    spectrogram(ax2,original_spec,frame_length,sr)

    #Filter Amplitude
    ax3 = fig.add_subplot(3,2,3)
    ax3.plot(np.linspace(0, sr//2, len(filter_property)//2),(np.abs(filter_property[:len(filter_property)//2])))
    ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.set(title="Filter Amplitude",xlabel = "Frequency [Hz]",ylabel = "Sound Amplitude")

    #Filter Phase
    ax4 = fig.add_subplot(3,2,5)
    angle = np.unwrap(np.angle(filter_property))
    ax4.plot(np.linspace(0, sr//2, len(filter_property)//2), angle[:len(filter_property)//2])
    ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax4.set(title="Filter Phase",xlabel = "Frequency [Hz]",ylabel = "Angle[rad]")

    #Filtered Signal Spectrogram
    ax5 = fig.add_subplot(3,2,4)
    spectrogram(ax5,filtered_spec,frame_length,sr)

    sf.write("resynthesized.wav", filtered_signal, sr, subtype="PCM_16")

    plt.tight_layout()
    plt.show()
    plt.close()
