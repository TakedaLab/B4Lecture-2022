import optuna.integration.lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')


def main():
    # load preprocessed training data
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')

    # split into training and validation data (20% validation set)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, Y_train,
        test_size=0.20,
        random_state=2,
    )

    # lightGBM
    # set training data
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

    # LightGBM parameters
    params = {
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "num_class": 10,
            "metric": "multi_logloss",
            "verbosity": -1,
    }

    # learn Models
    # parameters are automatically optimized by Optune
    evals_result = {}
    model = lgb.train(params,  # parameters
                      lgb_train,  # set training data
                      valid_sets=[lgb_train, lgb_eval],  # set validation data
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=20,  # if Log loss does not improve 20 times, training is terminated
                      evals_result=evals_result
                      )

    # estimate validation data (return estimation classes (0~9))
    y_pred_prob = model.predict(X_validation)
    # the class with the highest estimation probability -> the estimation class
    y_pred = np.argmax(y_pred_prob, axis=1)

    # display true and estimated values
    df_pred = pd.DataFrame({'true value': y_validation, 'estimated value': y_pred})

    print("Display true and estimated labels")
    print("---------------------------------")
    print(df_pred)

    # display true values and estimation probability
    df_pred_prob = pd.DataFrame({'true label': y_validation,
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

    print("Display true labels and predicted probabilities")
    print("-----------------------------------------------")
    print(df_pred_prob)

    # evaluate models
    acc = accuracy_score(y_validation, y_pred)
    print('Acc :', acc)

    # save the trained model as a pickle file
    filename = 'model/model_acc' + str(math.floor(acc * 100000)) + '.pickle'
    with open(filename, mode='wb') as f:
        pickle.dump(model, f)  # serialize objects

    # visualize of the training process
    plt.plot(evals_result['train']['multi_logloss'], label='train')
    plt.plot(evals_result['valid']['multi_logloss'], label='valid')
    plt.ylabel('Log loss')
    plt.xlabel('Boosting round')
    plt.title(f'Training performance \n Acc: {acc}')
    plt.legend()
    plt.show()
    plt.savefig("train_history.png")

    # get parameters optimized by Optune
    best_params = model.params
    print("  Params: ")
    print(best_params)


if __name__ == "__main__":
    main()
