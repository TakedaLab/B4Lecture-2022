import argparse
import time
from functools import wraps
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        process_time = round(time.time() - start, 4)
        print("--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--")
        print(f"It took {process_time} sec to process {func.__name__}")
        print("--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--")
        return result
    return wrapper


# output[0]から順に引数として渡す
def forward_algorithm(k, PI, A, B):
    """forward algorithm

    Args:
        k (np.ndarray): 出力系列. shape = (n, ).
        PI (np.ndarray): 初期確率. shape = (h, c, 1).
        A (np.ndarray): 状態遷移確率行列. shape = (h, c, c).
        B (np.ndarray): 出力確率. shape = (h, c, m).

    Returns:
        int: 出力系列kのモデル予測結果

    Note:
        n: 観測回数
        h: HMMのモデル数
        c: 状態数
        m: 出力記号の数
    """
    n = len(k)
    model = PI.shape[0]
    c = PI.shape[1]
    alpha = np.zeros((model, n, c))
    
    # 各モデルについて計算
    for h in range(model):
        # 初期化
        alpha[h, 0, :] = PI[h, :, 0] * B[h, :, 0]  # (model, n, c)
        print(alpha.shape)
        
        # 再帰的計算
        for t in range(1, n):
            alpha[h, t, :] = np.sum(alpha[h, t - 1, :] * A[h].T, axis = 1) * B[h, :, k[t]]
    
    # 確率の計算
    probability = np.sum(alpha, axis = 2)  # (model, n)
    predict = np.argmax(probability[:, -1])
    
    return predict


@stop_watch
def run_forward_algorithm(output, PI, A, B):
    """run forward algorithm

    Args:
        output (np.ndarray): 全出力系列. shape = (k, n).
        PI (np.ndarray): 初期確率. shape = (h, c, 1).
        A (np.ndarray): 状態遷移確率行列. shape = (h, c, c).
        B (np.ndarray): 出力確率. shape = (h, c, m).

    Returns:
        np.adarray: 予測結果

    Note:
        k: 出力系列数
        n: 観測回数
        h: HMMのモデル数
        c: 状態数
        m: 出力記号の数
    """
    predict = np.zeros(len(output))
    
    for k in range(len(output)):
        predict[k] = forward_algorithm(output[k], PI, A, B)    
    
    return predict


def viterbi_algorithm():
    return 0


def main():
    # process args
    parser = argparse.ArgumentParser(description="HMM Model prediction.")
    parser.add_argument("fname", type=str, help="Load filename (.pickle)")
    args = parser.parse_args()

    # get file name from command line
    fname = args.fname

    # load pickle data
    data = pickle.load(open(fname + ".pickle", "rb"))
    # data
    # ├─answer_models # 出力系列を生成したモデル（正解ラベル）
    # ├─output # 出力系列
    # └─models # 定義済みHMM
    #   ├─PI # 初期確率
    #   ├─A # 状態遷移確率行列
    #   └─B # 出力確率

    answer_models = np.array(data["answer_models"])
    output = np.array(data["output"])
    PI = np.array(data["models"]["PI"])
    A = np.array(data["models"]["A"])
    B = np.array(data["models"]["B"])

    '''
    print("answer_models")
    print(answer_models)
    print(answer_models.shape)
    print(type(answer_models))
    print("output")
    print(output)
    print(output.shape)
    print(type(output))
    print("PI")
    print(PI)
    print(PI.shape)
    print("A")
    print(A)
    print(A.shape)
    print("B")
    print(B)
    print(B.shape)
    '''
    
    predict = run_forward_algorithm(output, PI, A, B)
    print(predict)


if __name__ == "__main__":
    main()
