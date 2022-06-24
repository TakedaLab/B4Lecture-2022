import numpy as np
import pandas as pd
import time
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix


def forward(output, PI, A, B):
    """forward algorithm

    m: model number
    s: state number
    o: output number

    Args:
        output (ndarray (data_size, output_size)): output data
        PI (ndarray (m, s, 1)): Initial probability
        A (ndarray (m, s, s)): state transition probability matrix
        B (ndarray (m, s, o)): output probability

    Returns:
        ndarray (data_size, ): HMM predict
    """
    data_size, out_size = output.shape
    predict = np.zeros(data_size)

    # ------single loop----------
    start = time.time()
    # (m, s, 1)*(m, s, data_size)->(m, s, data_size)
    alpha_test = PI[:, :, 0, np.newaxis] * B[:, :, output[:, 0]]
    # (m, s, data_size)->(data_size, s, m)
    alpha_test = alpha_test.transpose(2, 1, 0)

    for j in range(1, out_size):
        # ((data_size, s, m)->(data_size, m, s))
        # *
        # ((m, s, data_size)->(data_size, m, s))
        alpha_test = np.sum(
            # (1, s, s, m) * (data_size, 1, s, m) ->(data_size, s, s, m)
            A.T[np.newaxis, :, :, :] * alpha_test[:, np.newaxis, :, :],
            axis=2,
        ).transpose(0, 2, 1) * B[:, :, output[:, j]].transpose(2, 0, 1)
        # (data_size, s, m)
        alpha_test = alpha_test.transpose(0, 2, 1)
    predict_test = np.argmax(np.sum(alpha_test, axis=1), axis=1)

    fin1 = time.time() - start

    # ------double loop----------
    start = time.time()
    for i in range(data_size):
        alpha = PI[:, :, 0] * B[:, :, output[i, 0]]
        for j in range(1, out_size):
            alpha = np.sum(A.T * alpha.T, axis=1).T * B[:, :, output[i, j]]
        predict[i] = np.argmax(np.sum(alpha, axis=1))
    fin2 = time.time() - start

    print(f"single loop (forward):{fin1}")
    print(f"double loop (forward):{fin2}")
    print(f"double / single (forward) ={fin2/fin1:.3f}")
    return predict_test


def viterbi(output, PI, A, B):
    """viterbi algorithm

    m: model number
    s: state number
    o: output number

    Args:
        output (ndarray (data_size, output_size)): output data
        PI (ndarray (m, s, 1)): Initial probability
        A (ndarray (m, s, s)): state transition probability matrix
        B (ndarray (m, s, o)): output probability

    Returns:
        ndarray (data_size, ): HMM predict
    """
    data_size, out_size = output.shape
    predict = np.zeros(data_size)

    # ------single loop----------
    start = time.time()
    # (m, s, 1)*(m, s, data_size)->(m, s, data_size)
    alpha_test = PI[:, :, 0, np.newaxis] * B[:, :, output[:, 0]]
    # (m, s, data_size)->(data_size, s, m)
    alpha_test = alpha_test.transpose(2, 1, 0)

    for j in range(1, out_size):
        # ((data_size, s, m)->(data_size, m, s))
        # *
        # ((m, s, data_size)->(data_size, m, s))
        alpha_test = np.max(
            # (1, s, s, m) * (data_size, 1, s, m) ->(data_size, s, s, m)
            A.T[np.newaxis, :, :, :] * alpha_test[:, np.newaxis, :, :],
            axis=2,
        ).transpose(0, 2, 1) * B[:, :, output[:, j]].transpose(2, 0, 1)
        # (data_size, s, m)
        alpha_test = alpha_test.transpose(0, 2, 1)
    predict_test = np.argmax(np.max(alpha_test, axis=1), axis=1)

    fin1 = time.time() - start

    # ------double loop----------
    start = time.time()

    for i in range(data_size):
        alpha = PI[:, :, 0] * B[:, :, output[i, 0]]
        for j in range(1, out_size):
            alpha = np.max(A.T * alpha.T, axis=1).T * B[:, :, output[i, j]]
        predict[i] = np.argmax(np.max(alpha, axis=1))

    fin2 = time.time() - start

    print(f"single loop (viterbi):{fin1}")
    print(f"double loop (viterbi):{fin2}")
    print(f"double / single (viterbi) ={fin2/fin1:.3f}")
    return predict_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="data filename .pickle")
    args = parser.parse_args()

    data = pickle.load(open(args.filename + ".pickle", "rb"))
    answer_models = data["answer_models"]  # 出力系列を生成したモデル
    output = np.array(data["output"])  # 出力系列
    PI = np.array(data["models"]["PI"])  # 初期確率
    A = np.array(data["models"]["A"])  # 状態遷移確率行列
    B = np.array(data["models"]["B"])  # 出力確率

    f_pre = forward(output, PI, A, B)
    v_pre = viterbi(output, PI, A, B)

    # plot
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(121)
    label = list(map(lambda x: x + 1, list(set(answer_models))))
    cm = confusion_matrix(answer_models, f_pre)
    cm = pd.DataFrame(cm, columns=label, index=label)
    acc = accuracy_score(answer_models, f_pre) * 100
    sns.heatmap(cm, annot=True, cbar=False, square=True, cmap="binary")
    plt.title(f"{args.filename} forward\n(Acc:{acc:.1f}%)")
    plt.xlabel("Prediction")
    plt.ylabel("Answer")

    plt.subplot(122)
    label = list(map(lambda x: x + 1, list(set(answer_models))))
    cm = confusion_matrix(answer_models, v_pre)
    cm = pd.DataFrame(cm, columns=label, index=label)
    acc = accuracy_score(answer_models, v_pre) * 100
    sns.heatmap(cm, annot=True, cbar=False, square=True, cmap="binary")
    plt.title(f"{args.filename} viterbi\n(Acc:{acc:.1f}%)")
    plt.xlabel("Prediction")
    plt.ylabel("Answer")

    plt.savefig(f"{args.filename}.png")
    plt.show()


if __name__ == "__main__":
    main()
