import librosa
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def feature_extraction(path_list):
    """
    Extract features using a list of wav files and return them as a list.
    The feature handled is the average of 13 MFCC dimensions (not including zero-order).

    Args:
        path_list(np.ndarray): path list of files from which to extract features.
    Returns:
        np.ndarray: extracted features.
    """

    load_data = (lambda path: librosa.load(path)[0])

    # read wav data using the file list and stores them in the list
    data = list(map(load_data, path_list))

    # extract the average of 13 MFCC dimensions as features
    features = np.array([np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data])

    return features


def main():

    # test data
    test = pd.read_csv("test_truth.csv")
    X_test = feature_extraction(test["path"].values)
    Y_test = np.array(test["label"].values)

    # load trained model
    with open('model/model_acc95185.pickle', mode='rb') as f:
        model = pickle.load(f)  # serialize objects

    # classify test data
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # display true and estimated values
    df_pred = pd.DataFrame({'true value': Y_test, 'predicted value': y_pred})
    print("Display true and predicted values")
    print("---------------------------------")
    print(df_pred)
    print("---------------------------------")

    # display true values and estimation probability
    df_pred_prob = pd.DataFrame({'true label': Y_test,
                                 'P(0)': y_pred_prob[:, 0],
                                 'P(1)': y_pred_prob[:, 1],
                                 'P(2)': y_pred_prob[:, 2],
                                 'P(3)': y_pred_prob[:, 3],
                                 'P(4)': y_pred_prob[:, 4],
                                 'P(5)': y_pred_prob[:, 5],
                                 'P(6)': y_pred_prob[:, 6],
                                 'P(7)': y_pred_prob[:, 7],
                                 'P(8)': y_pred_prob[:, 8],
                                 'P(9)': y_pred_prob[:, 9]
                                 })

    print("Display true values and predicted probabilities")
    print("-----------------------------------------------")
    print(df_pred_prob)
    print("-----------------------------------------------")

    # # evaluate models
    acc = accuracy_score(Y_test, y_pred) * 100
    print('Acc :', acc, '%')

    # plot confusion matrix
    plt.figure()
    cm_fwd = confusion_matrix(y_pred, Y_test)
    sns.heatmap(cm_fwd, cmap='Blues', square=True)
    plt.title(f"Result \n (Acc. {acc:.5f}%)")
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
