import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

class PCA:
    def __init__(self, data):
        self.data = data
        self.length = data.shape[0]
        self.deg = data.shape[1]


    def fit(self, x):







def plot2d(data, title):
    data_size = data.shape[1]
    x = data[:, 0 : data_size - 1]
    y = data[:, data_size - 1 :]

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, label="Observed Value")
    x1 = np.linspace(x.min(), x.max())
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title)
    ax.grid()
    ax.legend()
    plt.show()
    plt.close()



def plot3d(data, title):
    data_size = data.shape[1]
    x = data[:, 0 : data_size - 1]
    y = data[:, data_size - 1 :]

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(x[:, 0], x[:, 1], y, c="r", label="Observed data")
    x1 = np.linspace(x.min(), x.max())
    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$", title=title)
    ax.grid()
    ax.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":

    data1 = pd.read_csv('data1.csv').values
    data2 = pd.read_csv('data2.csv').values
    data3 = pd.read_csv('data3.csv').values

    model1 = PCA(data1)
    plot2d(data1, "data1")
    plot3d(data2, "data2")
