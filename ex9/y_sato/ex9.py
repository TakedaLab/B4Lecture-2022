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
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
    mfcc = np.array([librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13) for data in load_data])
    delta = np.array([librosa.feature.delta(x) for x in mfcc])
    feature = np.append(mfcc, delta, axis=1)

    return lenmax, feature


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


def plt_history(history, EPOCH):
        # 学習過程の可視化
    plt.plot(range(1, EPOCH+1), history.history['accuracy'], label="training")
    plt.plot(range(1, EPOCH+1), history.history['val_accuracy'], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("figs/" + "process_accuracy.png")
    plt.close()

    plt.plot(range(1, EPOCH+1), history.history['loss'], label="training")
    plt.plot(range(1, EPOCH+1), history.history['val_loss'], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("figs/" + "process_loss.png")
    plt.close()


def test_cm(l_train):
    test = pd.read_csv("test_truth.csv")
    _, X_test = feature_extract(test["path"].values, l_train)
    Y_test = np.array(test["label"].values)

    X_test = X_test[:, :, :, np.newaxis]

    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)

    cm = confusion_matrix(predict, Y_test)
    acc = np.sum(predict == Y_test) / predict.shape[0] * 100

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, cmap="Blues", annot=True, cbar=False, square=True)
    ax.set(
        xlabel="Ground Truth", ylabel="Predicted", title=f"Result\n(Acc:{acc}%\n"
    )

    plt.tight_layout()
    plt.savefig("figs/" +"cm.png")
    plt.show()
    plt.close()
    

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
    
    if args.use_trained_model:
        model = load_model(args.use_trained_model)
        model.summary()
        
    else:
        EPOCH = 30
        model = cnn(X_train.shape, output_dim=10)
        model.compile(loss="categorical_crossentropy",
                        optimizer=Adam(lr=1e-4),
                        metrics=["accuracy"])
        
        result = model.fit(X_train, 
                    Y_train,
                    batch_size=32, 
                    epochs=EPOCH,
                    verbose=1,
                    validation_data=(X_validation, Y_validation))
        model.save("leras_model/my_model_5.h5")

        plt_history(result, EPOCH)

        score = model.evaluate(X_validation, Y_validation, verbose=0)
        print("Validation accuracy:", np.round(score[1], decimals=5))

    test_cm(l_train)