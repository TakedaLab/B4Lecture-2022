import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from functools import wraps
import time


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


def gaussian(x, mu, sigma):
    """gaussian distribution

    Args:
        x (np.ndarray): input data. shape = (dim, ).
        mu (np.ndarray): means of distribution. shape = (dim, ).
        sigma (np.ndarray): variances of distribution. shape = (dim, dim).

    Returns:
        float: gaussian distribution
    """
    dim = len(x)
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)  # shape = (dim, dim)
    a = ((2 * np.pi) ** (0.5 * dim)) * (sigma_det ** 0.5)
    b = - 0.5 * (x - mu) @ sigma_inv @ (x - mu).T
    gaussian = np.exp(b) / a
    return gaussian


def calc_gaussian(input, mu, sigma):
    """calculate gaussian distribution

    Args:
        input (np.ndarray): input data. shape = (n, dim).
        mu (np.ndarray): means of distribution. shape = (dim, ).
        sigma (np.ndarray): variances of distribution. shape = (dim, dim).

    Returns:
        np.ndarray: gaussian distribution. shape = (n, ).
    """
    n = len(input)

    gaussians = np.zeros(n)
    for i in range(n):
        gaussians[i] = gaussian(input[i], mu, sigma)

    return gaussians


def calc_mixture_gaussian(input, pi, mu, sigma):
    """calculate mixture gaussian distribution

    Args:
        input (np.ndarray): input data. shape = (n, dim).
        pi (np.ndarray): mixing coefficients of each class. shape = (k, ).
        mu (np.ndarray): means of distributions. shape = (k, dim).
        sigma (np.ndarray): variances of distributions. shape = (k, dim, dim).

    Returns:
        np.ndarray: mixture gaussian distribution. shape = (k, n).
    """
    n = len(input)
    k = len(pi)

    mix_gaussian = np.zeros((k, n))
    for i in range(k):
        mix_gaussian[i] = pi[i] * calc_gaussian(input, mu[i], sigma[i])

    return mix_gaussian


def calc_loglikelihood(input, pi, mu, sigma):
    """calculate log-likelihood

    Args:
        input (np.ndarray): input data. shape = (n, dim).
        pi (np.ndarray): mixing coefficients of each class. shape = (k, ).
        mu (np.ndarray): means of distributions. shape = (k, dim).
        sigma (np.ndarray): variances of distributions. shape = (k, dim, dim).

    Returns:
        float: log-likelihood
    """
    mix_gaussian = calc_mixture_gaussian(input, pi, mu, sigma)  # shape = (k, n)
    sum_mix_gaussian = np.sum(mix_gaussian, axis=0)  # shape = (n, )
    loglikelihood = np.sum(np.log(sum_mix_gaussian))

    return loglikelihood


@stop_watch
def em_algorithm(input, pi, mu, sigma, epsilon=0.001, max_iter=50):
    """calculate EM-algorithm

    Args:
        input (np.ndarray): input data. shape = (n, dim).
        pi (np.ndarray): mixing coefficients of each class. shape = (k, ).
        mu (np.ndarray): means of distributions. shape = (k, dim).
        sigma (np.ndarray): variances of distributions. shape = (k, dim, dim).
        epsilon (float, optional): convergence condition. Defaults to 0.001.
        max_iter (int, optional): max iteration number. Defaults to 50.

    Returns:
        np.ndarray: log-likelihoods.
        np.ndarray: updated mixing coefficients of each class.
        np.ndarray: updated means of distributions.
        np.ndarray: updeted variances of distributions.
    """
    n = len(input)
    loglikelihood = []
    loglikelihood.append(calc_loglikelihood(input, pi, mu, sigma))

    for i in range(max_iter):
        # E step
        mix_gaussian = calc_mixture_gaussian(input, pi, mu, sigma)  # shape = (k, n)
        sum_mix_gaussian = np.sum(mix_gaussian, axis=0)  # shape = (n, )
        gamma = mix_gaussian / sum_mix_gaussian[np.newaxis, :]  # (k, n) <- (k, n) / (1, n)

        # M step
        # update n_k
        n_k = np.sum(gamma, axis=1)  # shape = (k, )
        # update pi
        pi = n_k / n

        # update mu
        mu = (gamma @ input) / n_k[:, np.newaxis]  # shape = (k, dim)

        # update sigma
        for k in range(len(n_k)):
            sigma[k] = 0
            for j in range(n):
                sigma[k] += gamma[k][j] * (input[j] - mu[k])[:, np.newaxis] @ (input[j] - mu[k])[np.newaxis, :]
        sigma /= n_k[:, np.newaxis, np.newaxis]  # (k, dim, dim) <- (k, dim, dim) / (k, 1, 1)

        new_loglikelihood = calc_loglikelihood(input, pi, mu, sigma)
        loglikelihood.append(new_loglikelihood)

        # convergence test
        if loglikelihood[i + 1] - loglikelihood[i] < epsilon:
            break

    return loglikelihood, pi, mu, sigma


@stop_watch
def k_means(input, k, max_iter=500):
    """k-means algorithm

    Args:
        X (np.ndarray): input data
        k (int): number of cluster
        max_iter (int, optional): maximum of iteration. Defaults to 300.

    Returns:
        np.ndarray: the data of clusters
        np.ndarray: the data of centroids
    """
    input_size, n_features = input.shape

    # randomly initialize initial centroids
    centroids = input[np.random.choice(input_size, k)]
    # array for the new centroids
    new_centroids = np.zeros((k, n_features))
    # array to store the cluster information to which each data belongs
    clusters = np.zeros(input_size)

    for _ in range(max_iter):
        # loop for all input data
        for i in range(input_size):
            # calculate the distance to each centroid from the data
            distances = np.sum((centroids - input[i]) ** 2, axis=1)
            # update the cluster based on distances
            clusters[i] = np.argsort(distances)[0]

        # recalculate centroid for all clusters
        for j in range(k):
            new_centroids[j] = input[clusters == j].mean(axis=0)

        # break if centrois has not changed
        if np.sum(new_centroids == centroids) == k:
            break
        centroids = new_centroids

    return centroids


def main():
    fname = args.fname
    k = args.mixed_number

    # read csv files as DataFrame
    df_data = pd.read_csv(fname + ".csv", header=None)

    # convert DataFrame form to array form
    data = df_data.to_numpy()

    # set drawing area
    plt.rcParams["figure.figsize"] = (8, 10)

    # fitting data
    # determine initial parameters
    n = data.shape[0]
    dim = data.shape[1]
    pi = np.full(k, 1 / k)
    # mu = np.random.randn(k, dim)
    mu = k_means(data, k)
    sigma = np.array([np.eye(dim) for _ in range(k)])

    # run em-algorithm & update parameters
    loglikelihood, pi, mu, sigma = em_algorithm(data, pi, mu, sigma)

    print("--------------------------------------------------------------")
    print("EM algorithm was iterated %i times" % len(loglikelihood))
    print("--------------------------------------------------------------")
    print("π　(mixed number) is: ")
    print(pi)
    print("")
    print("μ (means of distributions) is: ")
    print(mu)
    print("")
    print("Σ (variances of distributions) is: ")
    print(sigma)
    print("--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--")
    print("")

    if dim == 1:
        # set drawing area for data1
        fig, ax = plt.subplots(2, 1)
        # plot log-likelihood
        ax[0].plot(np.arange(0, len(loglikelihood), 1), loglikelihood, c='royalblue')
        ax[0].set(title="data1 log-likelihood", xlabel="iteration number", ylabel="log-likelihood")
        # plot scatter & mixture gaussian distribution
        x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), n)
        pdfs = np.zeros((n, k))
        for i in range(k):
            pdfs[:, i] = pi[i] * norm.pdf(x, loc=mu[i], scale=sigma[i])
        ax[1].scatter(data, np.zeros(len(data)), facecolor="None", edgecolors="royalblue", label="data1 sample")
        ax[1].scatter(mu, np.zeros(len(mu)), s=100, marker="*", c="r", label="centroids")
        ax[1].plot(x, np.sum(pdfs, axis=1), label="GMM", c='lime')
        ax[1].set(title="GMM data1", xlabel="data1", ylabel="mixture gaussian distribution")
        ax[1].legend()

        fig.tight_layout()
        fig.savefig("data1.png")

    if dim == 2:
        # set drawing area for data2&3
        fig, ax = plt.subplots(2, 1)
        ax[0] = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        ax[1] = plt.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=3)
        # plot log-likelihood
        ax[0].plot(np.arange(0, len(loglikelihood), 1), loglikelihood, c='royalblue')
        ax[0].set(title=fname + " log-likelihood", xlabel="iteration number", ylabel="log-likelihood")
        # plot scatter & mixture gaussian distribution
        x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), n)
        y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), n)
        X, Y = np.meshgrid(x, y)
        lines = np.dstack((X, Y))
        Z = np.zeros((n, n))
        for i in range(n):
            Z[i] = np.array([np.sum(calc_mixture_gaussian(lines[i, :], pi, mu, sigma), axis=0)])
        ax[1].scatter(mu[:, 0], mu[:, 1], s=100, marker="*", c="r", label="centroid")
        ax[1].contour(X, Y, Z, cmap="rainbow")
        ax[1].scatter(df_data[0], df_data[1], facecolor="None", edgecolors="royalblue", label=fname + " sample")
        ax[1].set(title=fname)
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(fname + ".png")


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="Fitting with GMM.")
    parser.add_argument("fname", type=str, help="Load filename (.csv)")
    parser.add_argument(
        "-k",
        "--mixed_number",
        type=int,
        help="mixed number of GMM (optional, Default=2)",
        default=2,
    )
    args = parser.parse_args()
    main()
