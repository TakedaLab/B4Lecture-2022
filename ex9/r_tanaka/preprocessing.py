import librosa
import pandas as pd
import numpy as np
from functools import partial
import warnings
warnings.filterwarnings('ignore')


# data augmentation: add white noise
def add_white_noise(x, rate=0.005):
    return x + rate*np.random.randn(len(x))


# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))


# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


def feature_extraction(path_list):
    """
    wavファイルのリストから特徴抽出を行い, リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """

    load_data = (lambda path: librosa.load(path)[0])

    # ファイルリストから該当のwavデータを読み込み、リストに格納する
    data = list(map(load_data, path_list))

    # data augmentation
    data_add_white_noise = list((map(add_white_noise, data)))
    data_stretch_sound = list(map(stretch_sound, data))
    mapfunc1 = partial(librosa.effects.pitch_shift, sr=22050, n_steps=1)
    data_pitch_shift_up1 = list(map(mapfunc1, data))
    mapfunc2 = partial(librosa.effects.pitch_shift, sr=22050, n_steps=-1)
    data_pitch_shift_down1 = list(map(mapfunc2, data))
    mapfunc3 = partial(librosa.effects.pitch_shift, sr=22050, n_steps=2)
    data_pitch_shift_up2 = list(map(mapfunc3, data))
    mapfunc4 = partial(librosa.effects.pitch_shift, sr=22050, n_steps=-2)
    data_pitch_shift_down2 = list(map(mapfunc4, data))

    # MFCC13次元の平均を特徴量として抽出
    features = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data])
    features_add_white_noise = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data_add_white_noise])
    features_stretch_sound = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data_stretch_sound])
    features_pitch_shift_up1 = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data_pitch_shift_up1])
    features_pitch_shift_down1 = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data_pitch_shift_down1])
    features_pitch_shift_up2 = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data_pitch_shift_up2])
    features_pitch_shift_down2 = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data_pitch_shift_down2])

    features = np.concatenate((features,
                               features_add_white_noise,
                               features_stretch_sound,
                               features_pitch_shift_up1,
                               features_pitch_shift_down1,
                               features_pitch_shift_up2,
                               features_pitch_shift_down2))

    return features


def main():
    # データの読み込み
    training = pd.read_csv("training.csv")
    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values)
    labels = np.array(training["label"].values)
    Y_train = np.concatenate((labels, labels, labels, labels, labels, labels, labels))

    np.save('X_train', X_train)
    np.save('Y_train', Y_train)


if __name__ == "__main__":
    main()
