import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def latex_for_polynomial_regression(deg, coef):
    """prepare a label in latex format for the graph of polynomial regression

    Args:
        deg (int): degree for multiple regression
        coef (ndarray): partial regression coefficients

    Returns:
        str: a label in latex format
    """
    formula = str("$y=")
    for i in range(deg):
        formula += (str(round(coef[i][0], 2))+"x^{"+str(i)+"}+")
    formula += (str(round(coef[i+1][0], 2))+"x^{"+str(i+1)+"}$")
    return formula


def latex_for_maltiple_regression(deg, coef):
    """prepare a label in latex format for the graph of maltiple regression

    Args:
        deg (int): degree for multiple regression
        coef (ndarray): partial regression coefficients

    Returns:
        str: a label in latex format
    """
    formula = str("$y=")
    w = 0
    for i in range(deg):
        for k in range(deg + 1 - i):
            formula += (str(round(coef[w][0], 2))+"x1^{"+str(i)+"}x2^{"+str(k)+"}+")
            w += 1
    i += 1
    for k in range(deg - i):
        formula += (str(round(coef[w][0], 2))+"x1^{"+str(i)+"x2^{"+str(k)+"}+")
        w += 1
    formula += (str(round(coef[w][0], 2))+"x1^{"+str(i)+"}x2^{"+str(k+1)+"}$")
    return formula


def plot_linear_regression(x_input, y_input, ax):
    """plot linear regression

    Args:
        x_input (ndarray): explanatory variable
        y_input (ndarray): response variable
        ax (list): list of axes objects
    """
    # find the regression coefficients
    a = ((x_input * y_input).mean() - (x_input.mean() * y_input.mean())) / ((x_input ** 2).mean() - x_input.mean() ** 2)
    b = -(a * x_input.mean()) + y_input.mean()

    # prepare x from -5 to 5 in increments of 0.1
    x = np.arange(-5, 5, 0.1)
    y = a * x + b

    # plot
    ax.plot(x, y, color='r', label="y={:.1f}x+{:.1f}".format(a, b))
    ax.legend()


def plot_polynomial_regression(x_input, y_input, deg, ax, alpha=1.0):
    """plot polynomial regression

    Args:
        x_input (ndarray): explanatory variable
        y_input (ndarray): response variable
        deg (int): degree for polynomial regression
        ax (list): list of axes objecs
        alpha(float): regularization term
    """
    # create array X
    j = np.arange(deg+1)
    # x_input[:,np.newaxis].shape==(datasize, 1)
    # j[np.newaxis,:].shape==(1, deg+1)
    # (datasize, deg+1) <- (datasize, 1) ** (1, deg+1)
    X = x_input[:, np.newaxis]**j[np.newaxis, :]

    # prepare for regularization
    alpha = float(alpha)
    eye = np.eye(X.shape[1])

    # vectorize y_input
    vec_y = y_input[:, np.newaxis]

    # find the partial regression coefficient
    coef = ((np.linalg.inv(X.T @ X + alpha * eye)) @ X.T) @ vec_y

    # prepare x as an equally spaced array from 0 to 10 with 100 elements
    x_axis = np.linspace(0, 10, 100)

    # find y from the partial regression coefficients
    j = np.arange(len(coef))
    y_axis = (x_axis[:, np.newaxis] ** j[np.newaxis, :]) @ coef

    # prepare labels for the graph
    formula = latex_for_polynomial_regression(deg, coef)

    # plot
    ax.plot(x_axis, y_axis, color='r', label=formula)
    ax.legend()


def multiple_regression(x_input1, x_input2, y_input, deg, ax, alpha=1.0):
    """plot multiple_regression

    Args:
        x_input1 (ndarray): explanatory variable
        x_input2 (ndarray): explanatory variable
        y_input (ndarray): response variable
        deg (int): degree for multiple regression
        ax (list): list of axes objecs
        alpha(float): regularization term
    """
    # vectorize y_input
    vec_y = y_input[:, np.newaxis]

    # create array X
    list_X = []
    for x1, x2 in zip(x_input1, x_input2):
        list = []
        for i in range(deg + 1):
            for k in range(deg + 1 - i):
                tmp = (x1 ** i) * (x2 ** k)
                list.append(tmp)
        list_X.append(list)
    X = np.array(list_X, dtype=float)

    # prepare for regularization
    alpha = float(alpha)
    eye = np.eye(X.shape[1])

    # find the partial regression coefficient
    coef = ((np.linalg.inv(X.T @ X + alpha * eye)) @ X.T) @ vec_y

    # generate mesh grid
    x1, x2 = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(0, 10, 100))

    # find y from the partial regression coefficients
    y = 0
    w = 0
    for i in range(deg + 1):
        for k in range(deg + 1 - i):
            y += coef[w] * (x1 ** i) * (x2 ** k)
            w += 1

    # prepare labels for the graph
    formula = latex_for_maltiple_regression(deg, coef)

    # plot
    ax.plot_wireframe(x1, x2, y, color='r', label=formula)
    ax.legend()


def main():
    # set drawing area for data1 and data2
    plt.rcParams["figure.figsize"] = (12, 10)
    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)

    # read csv files as DataFrame
    data1 = pd.read_csv("data1.csv")
    data2 = pd.read_csv("data2.csv")
    data3 = pd.read_csv("data3.csv")

    # draw a scatter plot of data1 and data2
    ax[0].scatter(data1['x1'], data1['x2'], label="observed")
    ax[0].set(title="data1", xlabel="x1", ylabel="x2")
    ax[0].legend()

    ax[1].scatter(data2['x1'], data2['x2'], label="observed")
    ax[1].set(title="data2", xlabel="x1", ylabel="x2")
    ax[1].legend()

    # set the data3 drawing area (3D)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    # set the elevation and azimuth of the axes in degrees
    ax3.view_init(elev=10, azim=230)

    # draw a scatter plot of data3
    ax3.plot(data3['x1'], data3['x2'], data3['x3'], marker="o", linestyle='None', label="observed")
    ax3.set(title="data3", xlabel="x1", ylabel="x2", zlabel="x3")
    ax3.legend()

    # change DataFrame form to array form
    data1_x1_array = data1["x1"].values
    data1_x2_array = data1["x2"].values

    data2_x1_array = data2["x1"].values
    data2_x2_array = data2["x2"].values

    data3_x1_array = data3["x1"].values
    data3_x2_array = data3["x2"].values
    data3_x3_array = data3["x3"].values

    # plot reguression
    plot_linear_regression(data1_x1_array, data1_x2_array, ax=ax[0])
    plot_polynomial_regression(data2_x1_array, data2_x2_array, deg=3, ax=ax[1], alpha=1.0)
    multiple_regression(data3_x1_array, data3_x2_array, data3_x3_array, deg=2, ax=ax3, alpha=1.0)

    # view and save all graphs
    plt.show()
    fig.savefig("data_1&2.png")
    fig3.savefig("data_3.png")


if __name__ == "__main__":
    main()
