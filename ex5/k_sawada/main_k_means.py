import argparse

import pandas as pd
import matplotlib.pyplot as plt

from my_functions import k_means


def main():
    parser = argparse.ArgumentParser(description='ex5')
    parser.add_argument("-i", "--input", help="input file id", type=int)
    args = parser.parse_args()
    
    # read data
    data = pd.read_csv(f"../data{args.input}.csv").values

    if (data.shape[1] == 2):  # 2d
        # plot initialize
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        ax = ax.flatten()
        
        # set clusters
        clusters = [2, 3, 4, 5]
        for c in range(len(clusters)):
            # classify
            labels = k_means.k_means(data, clusters[c])
            separated_data = k_means.data_separate(data, labels, clusters[c])
            
            # plot
            ax[c].set_xlabel("x1")
            ax[c].set_ylabel("x2")
            for i in range(clusters[c]):
                ax[c].scatter(separated_data[i][:, 0],
                              separated_data[i][:, 1])
    
    elif (data.shape[1] == 3):  # 3d
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                               subplot_kw=dict(projection='3d'))
        ax = ax.flatten()
        
        # set clusters
        clusters = [2, 3, 4, 5]
        for c in range(len(clusters)):
            # classify
            labels = k_means.k_means(data, clusters[c])
            separated_data = k_means.data_separate(data, labels, clusters[c])
            
            # plot
            ax[c].set_xlabel("x1")
            ax[c].set_ylabel("x2")
            for i in range(clusters[c]):
                ax[c].scatter(separated_data[i][:, 0],
                              separated_data[i][:, 1],
                              separated_data[i][:, 2])
    # plt.savefig(f"data{args.input}.png")
    plt.show()


if __name__ == "__main__":
    main()
