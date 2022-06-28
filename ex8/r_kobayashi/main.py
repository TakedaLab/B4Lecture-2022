import argparse
import os
from tracemalloc import stop

import matplotlib.pyplot as plt
import numpy as np
import pickle
from functools import wraps
import time
import math
import decimal
from sklearn.metrics import confusion_matrix
import seaborn as sns

def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        process_time = round(time.time() - start, 4)
        print("--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--")
        print(f"It took {process_time} sec to process {func.__name__}")
        return result
    return wrapper


def round_num(x, degit):
    x_degits = math.floor(math.log10(abs(x)))
    x_rounded = decimal.Decimal(str(x)).quantize(decimal.Decimal(str(10 ** (x_degits - degit + 1))), rounding = "ROUND_HALF_UP")
    return x_rounded


class HMM:
    def __init__(self, a, b, pi, k, m):
        self.a = a
        self.b = b
        self.pi = pi
        self.k = k
        self.m = m

    @stop_watch
    def forward(self, output, samplesize):
        # alpha = np.zeros(samplesize)
        # 初期化
        alpha = self.b[:, output[:, 0]].T.reshape(samplesize, self.k, 1)
        for i in range(samplesize):
                alpha[i, :, :] = alpha[i, :, :] * self.pi
        # 再帰的計算
        for j in range(self.m):
            for i in range(samplesize):
                alpha[i, :, :] = (alpha[i, :, :].T @ self.a).T
            alpha = self.b[:, output[:, j]].T.reshape(100, 3, 1) * alpha
        # 確率計算
        P = np.zeros(samplesize)
        for i in range(samplesize):
            P[i] = np.sum(alpha[i, :, :])
        return P

    @stop_watch
    def viterbi(self, output, samplesize):
        # 初期化
        psi = self.b[:, output[:, 0]].T.reshape(100, 3, 1)
        for i in range(samplesize):
            psi[i, :, :] = psi[i, :, :] * self.pi
        # 再帰的計算
        max = np.zeros((samplesize, self.k, 1))
        PSI = np.zeros((samplesize, self.m, self.k))
        for j in range(self.m):
            for i in range(samplesize):
                for k in range(self.k):
                    # k: 次の状態
                    # PSI[k]: 次の状態がkとなる確率最大
                    tmp = psi[i, :, :] * self.a[:, k].reshape(3, 1)
                    max[i, k, :] = np.max(tmp)
                    PSI[i, j, k] = np.argmax(tmp)
                    psi[i, k, :] = self.b[int(PSI[i, j, k]), output[i, j]] * max[i, k]
        # 終了
        P = np.zeros(samplesize)
        for i in range(samplesize):
            P[i] = np.max(psi[i, :, :])
        return PSI, P
        # 系列の復元


def main(args):
    path = os.path.dirname(__file__)
    fname = os.path.join(path, "data", args.fname)
    graph_title = os.path.splitext(args.fname)[0]
    data = pickle.load(open(fname, "rb"))

    answer = np.array(data['answer_models'])
    a = np.array(data['models']['A'])
    b = np.array(data['models']['B'])
    pi = np.array(data['models']['PI'])
    output = np.array(data['output'])
    modelsize = pi.shape[0]
    k = pi.shape[1]
    m = output.shape[1]
    samplesize = output.shape[0]

    models = []
    P_forward = np.zeros((modelsize, samplesize))
    PSI = np.zeros((modelsize, samplesize, m, k))
    P_viterbi = np.zeros((modelsize, samplesize))
    for i in range(modelsize):
        models.append(HMM(a[i], b[i], pi[i], k, m))
        P_forward[i] = models[i].forward(output, samplesize)
        PSI[i], P_viterbi[i] = models[i].viterbi(output, samplesize)
    print(">>> print(PSI)")
    print(PSI)
    P_forward = P_forward.T
    P_viterbi = P_viterbi.T
    print(">>> print(P_viterbi)")
    print(P_viterbi)
    pred_forward = np.zeros(samplesize)
    pred_viterbi = np.zeros(samplesize)
    for i in range(samplesize):
        pred_forward[i] = np.argmax(P_forward[i, :])
        pred_viterbi[i] = np.argmax(P_viterbi[i, :])
    print(">>> print(pred_forward)")
    print(pred_forward)
    print(">>> print(pred_viterbi)")
    print(pred_viterbi)
    cm_forward = confusion_matrix(answer, pred_forward)
    cm_viterbi = confusion_matrix(answer, pred_viterbi)
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.subplots_adjust(wspace=0.5)
    sns.heatmap(cm_forward, annot=True, cmap='Greys', ax=ax[0])
    ax[0].set(xlabel="predict model", ylabel="answer model", title=f"{graph_title} Forward")
    sns.heatmap(cm_viterbi, annot=True, cmap='Greys', ax=ax[1])
    ax[1].set(xlabel="predict model", ylabel="answer model", title=f"{graph_title} Viterbi")
    plt.savefig(os.path.join(path, "figs", "result.png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    args = parser.parse_args()

    main(args)
