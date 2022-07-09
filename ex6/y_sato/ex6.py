import numpy as np
import scipy.stats as sp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

class PCA:
    def __init__(self, data):
        self.data = data
        self.length = data.shape[0]
        self.deg = data.shape[1]


    def fit(self):
        """principal component analysis

        Returns:
            self.eig_vectors(ndarray): sorted eigen vectors
            self.values: sorted eigen values
            self.c_rate: contribution rate
        """

        self.data = sp.stats.zscore(self.data)
        data = self.data

        #分散共分散行列
        ave = np.average(data, axis=0)
        v = (data - ave).T @ (data - ave) / self.length

        eig_values, eig_vectors = np.linalg.eig(v)

        index = np.argsort(eig_values)[::-1]
        eig_values = eig_values[index]
        eig_vectors = eig_vectors.T[index] #wip:ここがわからん
        self.eig_vectors = eig_vectors
        self.values = eig_values
        self.c_rate = eig_values / np.sum(eig_values)
        return


    def compress(self):
        data = self.data
        vectors = self.eig_vectors
        comp_data = data @ vectors.T
        self.comp_data = comp_data
        return comp_data


    def cc_rate(self):
        """Cumulative contribution rate

        Args:

        Returns:
            ndarray, axis=(dimension, ) : Cumulative contribution rate

        """

        c_rate = self.c_rate
        cc_rate = np.zeros(c_rate.shape[0] + 1)

        for i in range(self.deg):
            cc_rate[i + 1] = c_rate[i] + cc_rate[i]

        return cc_rate


    def plot2d(self, ax, title):
        self.scatter2d(self.data, ax, title)
        self.plot_vector2d(self.data, ax)


    def scatter2d(self, data, ax, title):
        data_size = data.shape[1]
        x = data[:, 0]
        y = data[:, 1]
        #plot
        ax.scatter(x, y, color="black", label="Observed Value")
        ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title)


    def plot_vector2d(self, data, ax):
        data = data
        eig_vectors = self.eig_vectors

        data_size = data.shape[1]
        x = data[:, 0]
        y = data[:, 1]

        #plot
        x1 = np.linspace(x.min(), x.max())

        a2 = eig_vectors[:, 1] / eig_vectors[:, 0]
        ax.plot(x1, a2[0] * x1 , label=f"Contribution rate : {self.c_rate[0]:.3f}")
        ax.plot(x1, a2[1] * x1 , label=f"Contribution rate : {self.c_rate[1]:.3f}")



    def plot3d(self, ax, title):
        data = self.data
        eig_vectors = self.eig_vectors

        data_size = data.shape[1]
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        #plot
        ax.scatter3D(x, y, z, color="black", label="Observed data")

        x1 = np.linspace(x.min(), x.max())
        y1 = np.linspace(y.min(), y.max())
        X, Y = np.meshgrid(np.arange(x.min(), x.max(), 0.1), np.arange(y.min(), y.max(), 0.1))
        a2 = eig_vectors[:, 1] / eig_vectors[:, 0]
        a3 = eig_vectors[:, 2] / eig_vectors[:, 0]

        ax.plot(x1, a2[0] * x1, a3[0] * x1, label=f"Contribution rate : {self.c_rate[0]:.3f}")
        ax.plot(x1, a2[1] * x1, a3[1] * x1, label=f"Contribution rate : {self.c_rate[1]:.3f}")
        ax.plot(x1, a2[2] * x1, a3[2] * x1, label=f"Contribution rate : {self.c_rate[2]:.3f}")

        ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=title)
        ax.grid()
        ax.legend()



if __name__ == "__main__":

    data1 = pd.read_csv('data1.csv').values
    data2 = pd.read_csv('data2.csv').values
    data3 = pd.read_csv('data3.csv').values

    model1 = PCA(data1)
    model1.fit()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    model1.plot2d(ax1, "data1")
    plt.show()

    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection="3d")
    model2 = PCA(data2)
    model2.fit()
    model2.plot3d(ax2, "data2")
    plt.show()

    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    comp_data = model2.compress()
    model2.scatter2d(comp_data, ax3, "data2_2D")
    plt.show()

    fig = plt.figure()
    ax4 = fig.add_subplot(111)
    model3 = PCA(data3)
    model3.fit()
    cc_rate = model3.cc_rate()
    cc_min = np.min(np.where(cc_rate >= 0.9))
    x_line = np.arange(cc_rate.shape[0])
    y_line = np.linspace(min(cc_rate), max(cc_rate), cc_rate.shape[0])
    ax4.plot(x_line, cc_rate)
    ax4.plot(x_line, np.full(cc_rate.shape[0], cc_rate[cc_min]), linestyle="dashed", label=f'rate={cc_rate[cc_min]:.4f}')
    ax4.plot(np.full(cc_rate.shape[0], cc_min), y_line, linestyle="dashed", label=f'dim={cc_min}')
    ax4.set(xlabel="$x_1$", ylabel="$x_2$", title="data3 Contribution Rate")
    ax4.legend()
    plt.show()
