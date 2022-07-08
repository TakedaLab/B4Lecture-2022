import argparse

import librosa
import numpy as np
import pandas as pd
import csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import glob
import matplotlib.pyplot as plt
import time
import os


def cnn(input_shape, output_dim):
    """CNNモデルの構築
    Args:
        input_shape:
        output_dim:
    Returns:
        model: 定義済みモデル
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(input_shape[1], input_shape[2], 1), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.add(Activation("softmax"))
    model.summary()

    return model


def feature_extract(path_list, l):
    """wavファイルのリストから特徴抽出し、リストで返す
    Args:
        path_list(ndarray): file path list
    Returns:
        features(ndarray): feature value
    """

    sr = 8000
    load_data = [librosa.load(f"../{path}")[0] for path in path_list]
    
    lenmax = np.max([len(data) for data in load_data])

    #系列長を一定にするために後ろをゼロパディング
    if l >= lenmax:
        load_data = [np.pad(data, [0, l - len(data)], 'constant') for data in load_data]
    else:
        load_data = [np.pad(data, [0, lenmax - len(data)], 'constant') for data in load_data]
    print(np.array(load_data).shape)
    mfcc = np.array([librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13) for data in load_data])
    delta = np.array([librosa.feature.delta(x) for x in mfcc])
    feature = np.append(mfcc, delta, axis=1)

    return lenmax, feature


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.get_cmap("Blues")):
    """
    予測結果の混合行列をプロット
    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    """
    cm = confusion_matrix(predict, ground_truth)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.set(xlabel="Ground Truth", ylabel="Predicted")
    plt.savefig("cm.png")
    plt.show()


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する
    Args:
        paths: テストする音声ファイルリスト
        outputs:
    """
    with open("result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_to_truth", type=str, help='テストデータの正解ファイルCSVのパス')
    parser.add_argument("-u", "--use_trained_model", type=str, help='学習済みのモデルのパス')
    parser.add_argument("-a", "--add_train_data", type=str, help='学習データを追加する(yes)')
    args = parser.parse_args()

    train = pd.read_csv('../training.csv')
    test = pd.read_csv('../test.csv')
    
    l_train, X_train = feature_extract(train["path"], 0)
    _, X_test = feature_extract(test["path"], l_train)

    Y_train = np_utils.to_categorical(train["label"], num_classes=10)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train,
        test_size=0.2,
        random_state=20220708,
    )
    
    X_train = X_train[:, :, :, np.newaxis]
    X_validation = X_validation[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]

    print(X_train.shape)
    print(X_test.shape)
    
    if args.use_trained_model:
        model = load_model(args.use_trained_model)
        model.summary()

    else:
        model = cnn(X_train.shape, output_dim=10)
        model.compile(loss="categorical_crossentropy",
                        optimizer=Adam(lr=1e-4),
                        metrics=["accuracy"])
        
        model.fit(X_train, 
                Y_train,
                batch_size=32, 
                epochs=30,
                verbose=1)
        model.save("leras_model/my_model_5.h5")

        score = model.evaluate(X_validation, Y_validation, verbose=0)
        print("Validation accuracy:", np.round(score[1], decimals=5))

    predict = model.predict(X_test)
    predicted_values = np.argmax(predict, axis=1)
    
    write_result(test["path"].values, predicted_values)