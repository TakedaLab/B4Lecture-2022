import numpy as np


def k_means(data, clusters=3):
    """k-means clustering

    Args:
        data (ndarray, axis=(data count, each data)): input data
        clusters (int, optional): clusters count. Defaults to 3.

    Returns:
        ndarray: ndarray of cluster index
    """
    length = len(data)
    dimension = data.shape[1]
    
    # pick initial centroids
    # current implement is random
    centroids = np.random.randint(0, length, size=(clusters))
    centroids = data[centroids]

    distances = np.zeros((length, clusters))

    while True:
        # NOTE: for文を行列演算に置き換えれたら良いけど, 思いつかん
        # calculate distance between each data and centroids
        for i in range(length):
            distances[i] = np.sum(np.power(data[i] - centroids, 2), axis=1)
        # classify based on distance
        belongs = np.argmin(distances, axis=1)
        
        # update centroids
        centroids_new = np.zeros((clusters, dimension))
        for c in range(clusters):
            centroids_new[c] = np.average(data[belongs == c], axis=0)

        # judge convergence
        if np.all(centroids_new == centroids):
            break
        centroids = centroids_new
    
    return belongs


def data_separate(data, labels, clusters):
    """data separate based on labels

    Args:
        data (ndarray, axis=(data count, each data)): input data
        labels (ndarray, axis=data count): clustered label
        clusters (int): cluster count

    Returns:
        list of ndarray: separate with each clusters
    """
    separated = []
    for i in range(clusters):
        separated.append(data[labels == i])

    return separated
