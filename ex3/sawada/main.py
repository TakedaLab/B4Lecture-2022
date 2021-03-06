import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def regression_2d(data, title, dim=1,
                  reg_coef=0.0):
    """regression for 2d data

    Args:
        data (ndarray, axis=(data_column, 2)): observed data
        title (str): graph title
        dim (int, optional): polynominal degree. Defaults to 1.
        reg_coef (float, optional): regularize coefficient. Defaults to 0.0.
    """
    data_regression = np.zeros((data.shape[0], dim + 1))
    density = 50
    x_linspace = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), density)
    x_plot = np.zeros((density, dim))
    
    for i in range(dim):
        data_regression[:, i] = np.power(data[:, 0], i + 1)
        x_plot[:, i] = np.power(x_linspace, i + 1)
    data_regression[:, dim] = data[:, 1]

    coefficient = linear_regression(data_regression, reg_coef)
    # plot initialization
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(hspace=0.6)

    ax.set_title(title, fontsize=24)
    ax.set_xlabel("x1")
    ax.set_ylabel("y")
    ax.scatter(data[:, 0], data[:, 1], color="blue",
               label="Observed data")

    # regression result
    y_reg = np.concatenate([np.ones((len(x_plot), 1)), x_plot], 1) \
        @ coefficient

    label = "y = " + "{:.3g}".format(coefficient[0])
    for i in range(len(coefficient) - 1):
        label += " + " + "{:.3g}".format(coefficient[i + 1]) \
            + f"$x^{i+1}$"
    # label = "estimated"
    # print(coefficient)
    ax.plot(x_plot[:, 0], y_reg, color="red", label=label)
    ax.legend(fontsize=20)
    plt.savefig(f"{title}_{dim}th_polynomial.png")
    plt.show()

    # evaluete model
    y_true = data[:, 1]
    y_pred = np.concatenate([np.ones((len(data), 1)), data_regression[:, :-1]], 1) @ coefficient
    rsme = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"rsme: {rsme}")
    r2 = r2_score(y_true, y_pred)
    print(f"r2: {r2}")
        
        
def regression_3d(data, title, dim=1,
                  reg_coef=0.0):
    """regression for 3d data

    Args:
        data (ndarray, axis=(data_column, 3)): observed data
        title (str): graph title
        dim (int, optional): polynominal degree. Defaults to 1.
        reg_coef (float, optional): regularize coefficient. Defaults to 0.0.
    """
    density = 50
    data_regression = np.zeros(
        (data.shape[0], ((dim + 1) * (dim + 2)) // 2))
    x1_linspace = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), density)
    x2_linspace = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), density)
    x1, x2 = np.meshgrid(x1_linspace, x2_linspace)
    x_plot = np.zeros((density, density, ((dim + 1) * (dim + 2)) // 2 - 1))
    
    col = 0
    for i in range(dim):
        for j in range(i + 2):
            data_regression[:, col] \
                = np.power(data[:, 0], (i + 1 - j)) \
                * np.power(data[:, 1], (j))
            x_plot[:, :, col] \
                = np.power(x1, (i + 1 - j)) \
                * np.power(x2, (j))
            col += 1
    data_regression[:, -1] = data[:, -1]

    coefficient = linear_regression(data_regression, reg_coef)
    # plot initialization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(hspace=0.6)

    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('y')
    ax.scatter(data.T[0], data.T[1], data.T[2],
               color="blue", label="Ovserved data")
    # regression result
    y_reg = np.concatenate([np.ones((density, density, 1)), x_plot], 2) \
        @ coefficient
    label = "y = " + "{:.3g}".format(coefficient[0])
    col = 1
    for i in range(dim):
        for j in range(i + 2):
            label += " + " + "{:.3g}".format(coefficient[col]) \
                + f"$x_1^{i + 1 - j}x_2^{j}$"
            col += 1
    ax.plot_wireframe(x1, x2, y_reg, color="red", label=label)
    ax.legend(fontsize=14)
    plt.savefig(f"{title}_{dim}th_polynomial.png")
    plt.show()


def linear_regression(data, reg_coef=0.0):
    """linear regression

    Args:
        data (ndarray, size=(data_column, variables)): observation data.
        reg_coef (float, optional): regularize coefficient. Defaults to 0.0.

    Returns:
        ndarray: regression coefficient (intercept, slope, slope,...)
    """
    data_length = data.shape[0]
    data_dim = data.shape[1]

    # add constant term
    x = np.concatenate([np.ones((data_length, 1)), data[:, :-1]], 1)

    parameters = \
        np.linalg.inv(x.T @ x - reg_coef * np.identity(data_dim)) \
        @ x.T \
        @ data[:, -1]

    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ex3')
    parser.add_argument("-i", "--input", help="input file id", type=int)
    parser.add_argument("-d", '--dimension', type=int)
    parser.add_argument("-r", "--regularize", type=float)
    args = parser.parse_args()

    # read data
    data = pd.read_csv(f"../data{args.input}.csv").values

    if (data.shape[1] == 2):
        regression_2d(data, f"data{args.input}", dim=args.dimension,
                      reg_coef=args.regularize)
    if (data.shape[1] == 3):
        regression_3d(data, f"data{args.input}", dim=args.dimension,
                      reg_coef=args.regularize)
