import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pca(data, num_components=2):
    # mean Centering the data
    data_mean = data - np.mean(data, axis=0)

    # calculating the covariance matrix of the mean-centered data.
    cov_mat = np.cov(data_mean, rowvar=False)

    # calculating Eigenvalues and Eigenvectors of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    # similarly sort the eigenvectors
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Contribution Rate
    c_rate = sorted_eigenvalue[:] / np.sum(sorted_eigenvalue[:])

    # select the first n eigenvectors, n is desired dimension
    # of our final reduced data.
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # transform the data
    data_reduced = np.dot(eigenvector_subset.T, data_mean.T).T
    return data_reduced, sorted_eigenvectors, c_rate, sorted_eigenvalue


def main():
    # Argparse
    parser = argparse.ArgumentParser(description='Name of the Data File')
    parser.add_argument('-fn', metavar='-f', dest='filename', type=str, help='Enter the Audio File Name',
                        required=True)
    args = parser.parse_args()

    # read data
    data = pd.read_csv(args.filename, header=None)
    # Standardized Data
    data = (data - data.mean()) / data.std()  # For this case, is it better to normalized or standardized the data?
    data_pca, sorted_eigen, c_rate, sorted_eigen_val = pca(data, 2)
    if len(data.columns) == 2:
        data.columns = ['x1', 'x2']
        x1 = np.linspace(min(data['x1']), max(data['x1']), 100)
        coe_x2 = sorted_eigen[1] / sorted_eigen[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, title=args.filename, xlabel="x1", ylabel="x2", )
        ax.scatter(data['x1'], data['x2'])
        ax.plot(x1, coe_x2[0] * x1, color="m", label=f"Contribution rate: {c_rate[0]:.3}")
        plt.plot(x1, coe_x2[1] * x1, color="c", label=f"Contribution rate: {c_rate[1]:.3}")
        ax.legend(loc='best')
        plt.savefig("data1.png")
        plt.show()

    elif len(data.columns) == 3:
        data.columns = ['x1', 'x2', 'x3']
        # Fig 3D of Data2
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.scatter(data['x1'], data['x2'], data['x3'], color='m')
        x1 = np.linspace(min(data['x1']), max(data['x1']), 100)
        coeffx2 = sorted_eigen[1] / sorted_eigen[0]
        coeffx3 = sorted_eigen[2] / sorted_eigen[0]
        ax.plot(x1, coeffx2[0] * x1, coeffx3[0] * x1, color="r", label=f"Contribution rate: {c_rate[0]:.3}")
        ax.plot(x1, coeffx2[1] * x1, coeffx3[1] * x1, color="g", label=f"Contribution rate: {c_rate[1]:.3}")
        ax.plot(x1, coeffx2[2] * x1, coeffx3[2] * x1, color="c", label=f"Contribution rate: {c_rate[2]:.3}")
        ax.legend()
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        plt.savefig("data2.png")
        plt.show()
        plt.close()

        # dimension reduction data 2 to 2D
        plt.figure()
        plt.title("Dimension reduced data2")
        plt.scatter(data_pca[:, 0], data_pca[:, 1], color='m')
        plt.xlabel(f'PC1 {c_rate[0]:.3}', size=14)
        plt.ylabel(f'PC2 {c_rate[1]:.3}', size=14)
        plt.show()
        plt.savefig("data2_dim_reduction.png")
        plt.close()

    else:
        # Cumulative sum of contribution rate
        con_rate_sum = np.cumsum(c_rate)
        # point where the rate >0.9
        point = np.min(np.where(con_rate_sum >= 0.9))+1

        plt.figure()
        plt.title('cumulative contribution rate')
        plt.xlabel('dimension')
        plt.ylabel('cumulative contribution rate')
        x = np.arange(1, len(con_rate_sum)+1)
        plt.plot(x, con_rate_sum, color='m')
        plt.axhline(con_rate_sum[point-1], color='r', linestyle='--',
                    label=f'({point}, {con_rate_sum[point-1]:0.3f})')
        plt.axvline(point, color='r', linestyle='--')
        plt.xlim([0, 100])
        plt.ylim([min(con_rate_sum), max(con_rate_sum)])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('cumulative.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
