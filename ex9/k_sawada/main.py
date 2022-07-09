import pickle
import os
import datetime

import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from sklearn.model_selection import train_test_split
from keras.utils import np_utils 


SAMPLING_RATE = 8000  # by wave format data
LABELS = 10  # number of labels to classify
HIDDEN_UNITS = 128
# https://qiita.com/everylittle/items/ba821e93d275a421ca2b
DATA_LENGTH = 4096
SPLIT_SEED = 1


def get_length_info():
    """show max wav length in dataset
    """
    train_csv = pd.read_csv("../training.csv", dtype=str, encoding='utf8')
    test_csv = pd.read_csv("../test.csv", dtype=str, encoding='utf8')
    train_length = len(train_csv)
    count = train_length + len(test_csv)
    length = np.zeros(count)
    for i, row in train_csv.iterrows():
        length[i] = len(librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)[0])
    for i, row in test_csv.iterrows():
        length[i + train_length] = len(librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)[0])
    print(f"max_length: {np.max(length)} ({np.argmax(length)})")  # > 3000
    print(f"avg: {np.average(length)}")
    q75, q25 = np.percentile(length, [75 ,25])
    print(f"percentile: {q25} - {q75}")
    
    # path_list = sorted(glob.glob("../free-spoken-digit-dataset/recordings/*"))
    # count = len(path_list)
    # length = np.zeros(count)
    # for i in range(len(path_list)):
    #     length[i] = len(librosa.load(path_list[i], sr=SAMPLING_RATE, mono=True)[0])
    # print(f"max_length: {np.max(length)}")  # > 3000


def preview_data():
    train_csv = pd.read_csv("../training.csv", dtype=str, encoding='utf8')    
    print(train_csv)
    row = train_csv.iloc[0]
    print(row)
    wav, _ = librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav)
    plt.show()


def load_wav_train():
    """load train wave file

    Returns:
        ndarray, axis=(sample, time): wav
        ndarray, axis=(sample): answer label
    """
    if os.path.isfile('wav.pickle') and os.path.isfile('ans.pickle'):
        with open('wav.pickle', 'rb') as f:
            wav = pickle.load(f)
        with open('ans.pickle', 'rb') as f:
            label = pickle.load(f)
    else:
        train_csv = pd.read_csv("../training.csv", dtype=str, encoding='utf8')
        wav = np.zeros((len(train_csv), DATA_LENGTH))
        label = np.zeros(len(train_csv), dtype=np.int16)
        for i, row in train_csv.iterrows():
            wav_tmp, _ = librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)
            # zero padding at end of wav
            if (len(wav_tmp) > DATA_LENGTH):
                wav[i] = wav_tmp[0:DATA_LENGTH]
            else:
                wav[i, 0:len(wav_tmp)] = wav_tmp
            label[i] = row.label
        with open('wav.pickle', 'wb') as f:
            pickle.dump(wav, f)
        with open('ans.pickle', "wb") as f:
            pickle.dump(label, f)
    return wav, label


def expand_data(x, ans):
    """augument data by add latency

    Args:
        x (ndarray, axis=(data, ...)): input data
        ans (ndarray, axis=(data)): answer label
    Returns:
        ndarray, axis=(data, ...): expanded data
        ndarray, axis=(data): expanded answer label
    """
    latency = [0, 50, 100, 150, 200]
    x_shape = x.shape
    count = x_shape[0]
    series_length = x_shape[1]

    # new data array
    new_x = np.zeros_like(np.concatenate([x for i in range(len(latency))], axis=0))
    new_ans = np.concatenate([ans for i in range(len(latency))], axis=0)
    for i in range(len(latency)):
        new_x[i * count: (i + 1) * count, latency[i]: ] = x[:, 0: series_length - latency[i]]
    return new_x, new_ans


def wav2spectrogram(wav):
    _, _, specs = signal.spectrogram(wav)
    #_, _, specs = signal.spectrogram(wav, fs=SAMPLING_RATE, nperseg=128, nfft=2**11)
    return specs


def main():
    # read data
    x, answer_label = load_wav_train()
    # expand data
    x, answer_label = expand_data(x, answer_label)
    
    # split data for train and test
    x_train, x_test = train_test_split(x, random_state=SPLIT_SEED)
    ans_train, ans_test = train_test_split(answer_label, random_state=SPLIT_SEED)
    
    # one-hot
    ans_train = np_utils.to_categorical(ans_train)
    ans_test = np_utils.to_categorical(ans_test)

    # convert to a feature
    # x_train = wav2spectrogram(x_train)[:, :20, :]
    # x_test = wav2spectrogram(x_test)[:, :20, :]

    # convert to mfcc
    tmp_train = []
    tmp_test = []
    for i in range(len(x_train)):
        tmp_train.append(librosa.feature.mfcc(x_train[i]))
    for i in range(len(x_test)):
        tmp_test.append(librosa.feature.mfcc(x_test[i]))
    x_train = np.array(tmp_train)
    x_test = np.array(tmp_test)

    # preview
    if (False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(x_train[0])
        # plt.savefig("a.png")
    
    print(x_train.shape)
    input_shape = x_train.shape[1:]

    # create model
    model = models.Sequential()
    # https://www.tensorflow.org/guide/keras/masking_and_padding?hl=ja
    model.add(layers.Masking(input_shape=input_shape, mask_value=-1.0))
    model.add(layers.LSTM(HIDDEN_UNITS, input_shape=input_shape, return_sequences=False))
    model.add(layers.Dense(LABELS))
    model.add(layers.Activation("softmax"))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    fit_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss',
                                patience=5,
                                mode='min')
    ]

    # train
    epochs = 1000
    result = model.fit(x_train, ans_train,
                       epochs=epochs,
                       batch_size=512,
                       shuffle=True,
                       validation_data=(x_test, ans_test),
                       callbacks=fit_callbacks)

    # save model
    now = datetime.datetime.now()
    model.save_weights(f"keras_model/{now}model_weight.hdf5")
    model_arc_json = model.to_json()
    open(f"keras_model/{now}model_architecture.json", mode='w').write(model_arc_json)

    # output score
    score = model.evaluate(x_test, ans_test, verbose=0)
    print('test xentropy:', score)

    # plot train history
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"train history\nlast val_acc: {result.history['val_accuracy'][-1]:4f}")
    ax.plot(result.history["accuracy"], label="training")
    ax.plot(result.history["val_accuracy"], label="validation")
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy")
    ax.legend()
    plt.savefig(f"keras_model/{now}result.png")


if __name__ == '__main__':
    # get_length_info()
    # preview_data()
    main()
