import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import time
from functools import wraps
import sklearn.metrics as skl


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


class HMM:
    def __init__(self, k, m, a, b, pi):
        """ 
        Parameter
        ---------
        k : int
            number of status of latent variable
        m : int
            number of status of output variable
        a : np.ndarray(k, k) 
            state transition probability distribution
        b : np.ndarray(k,m)
            Output probability distribution
        pi : np.ndarray(k, 1)
            initial state probability distribution
        """
        self.k = k
        self.m = m
        self.a = a
        self.b = b
        self.pi = pi
        

    def forward(self, n_out, size_out, output):
        """ 
        Forward algorithm
        
        Parameter
        ---------
        n_out : int
            number of output series
        size_out : int
            size of output series
        output : np.ndarray(n_out, size_out)
            output series
        
        Return
        ------
        prob : np.ndarray(100,)

        """
        # 初期化
        alpha = self.b[:, output[:, 0]].T.reshape(n_out, self.k, 1)
        for i in range(n_out):
                alpha[i, :, :] = alpha[i, :, :] * self.pi
        # 再帰的計算
        for j in range(size_out):
            for i in range(n_out):
                alpha[i, :, :] = (alpha[i, :, :].T @ self.a).T
            alpha = self.b[:, output[:, j]].T.reshape(n_out, self.k, 1) * alpha
        # 確率計算
        prob = np.zeros(n_out)
        for i in range(n_out):
            prob[i] = np.sum(alpha[i, :, :])
        return prob


    def viterbi(self, n_out, size_out, output):
        """ 
        Viterbi algorithm
        
        Parameter
        ---------
        n_out : int
            number of output series
        size_out : int
            size of output series
        output : np.ndarray(n_out, size_out)
            output series
        
        Return
        ------
        prob : np.ndarray(100,)
         """
        # 初期化
        psi = self.b[:, output[:, 0]].T.reshape(n_out, self.k, 1)
        for i in range(n_out):
            psi[i, :, :] = psi[i, :, :] * self.pi
        # 再帰的計算
        max = np.zeros((n_out, self.k, 1))
        PSI = np.zeros((n_out, size_out, self.k))
        for j in range(size_out):
            for i in range(n_out):
                for k in range(self.k):
                    # k: 次の状態
                    # PSI[k]: 次の状態がkである確率が最大となる状態
                    tmp = psi[i, :, :] * self.a[:, k].reshape(self.k, 1)
                    max[i, k, :] = np.max(tmp)
                    PSI[i, j, k] = np.argmax(tmp)
                    psi[i, k, :] = self.b[int(PSI[i, j, k]), output[i, j]] * max[i, k]
        # 終了
        prob = np.zeros(n_out)
        for i in range(n_out):
            prob[i] = np.max(psi[i, :, :])
        return prob


    @stop_watch
    def run_forward(models, n_model, n_out, size_out, output):
        prob = np.zeros((n_model, n_out))
        for i in range(n_model):
            prob[i] = models[i].forward(n_out, size_out, output)
        return prob.T


    @stop_watch
    def run_viterbi(models, n_model, n_out, size_out, output):
        prob = np.zeros((n_model, n_out))
        for i in range(n_model):
            prob[i] = models[i].viterbi(n_out, size_out, output)
        return prob.T


def main(args):
    # データの読み込み
    path = os.path.dirname(__file__)
    fname = os.path.join(path, "data", args.fname)
    graph_title = os.path.splitext(args.fname)[0]
    data = pickle.load(open(fname, "rb"))
    answer = np.array(data['answer_models'])
    a = np.array(data['models']['A'])
    b = np.array(data['models']['B'])
    pi = np.array(data['models']['PI'])
    output = np.array(data['output'])
    """ print(a.shape)
    print(b.shape)
    print(pi.shape) """
    n_model = pi.shape[0]
    k = pi.shape[1]
    m = b.shape[1]
    n_out = output.shape[0]
    size_out = output.shape[1]
    # 確率の計算
    models = []
    prob_forward = np.zeros((n_model, n_out))
    prob_viterbi = np.zeros((n_model, n_out))
    for i in range(n_model):
        models.append(HMM(k, m, a[i], b[i], pi[i]))
    prob_forward = HMM.run_forward(models, n_model, n_out, size_out, output)
    prob_viterbi = HMM.run_viterbi(models, n_model, n_out, size_out, output)
    # 生成元のモデルを予測
    pred_forward = np.zeros(n_out)
    pred_viterbi = np.zeros(n_out)
    for i in range(n_out):
        pred_forward[i] = np.argmax(prob_forward[i, :])
        pred_viterbi[i] = np.argmax(prob_viterbi[i, :])
    # 結果のプロット
    cm_forward = skl.confusion_matrix(answer, pred_forward)
    acc_forward = skl.accuracy_score(answer, pred_forward)
    cm_viterbi = skl.confusion_matrix(answer, pred_viterbi)
    acc_viterbi = skl.accuracy_score(answer, pred_viterbi)
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.subplots_adjust(wspace=0.5)
    sns.heatmap(cm_forward, annot=True, cmap='Blues', ax=ax[0])
    ax[0].set(xlabel="predict model",
              ylabel="answer model",
              title=f"{graph_title} Forward\n(acc = {acc_forward})",)
    sns.heatmap(cm_viterbi, annot=True, cmap='Blues', ax=ax[1])
    ax[1].set(xlabel="predict model",
              ylabel="answer model",
              title=f"{graph_title} Viterbi\n(acc = {acc_viterbi})",)
    plt.savefig(os.path.join(path, "figs", graph_title+".png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    args = parser.parse_args()

    main(args)
