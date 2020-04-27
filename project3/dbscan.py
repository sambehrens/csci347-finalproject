from typing import List, Union, Dict

import numpy as np


def distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Gets the distance between two points.

    :param p: First point.
    :param q: Second point.
    :return: Distance.
    """
    return np.sqrt(np.sum((p - q) ** 2))


def get_neighbors(data, index, epsilon):
    """
    Gets indexes of points less than epsilon distance away from the
    point at index index.

    :param data: Array of points.
    :param index: Index of the point to get the neighbors of.
    :param epsilon: Distance away from the point.
    :return: Indexes of the neighbors.
    """
    distances = np.sqrt(np.sum((data - data[index]) ** 2, axis=1))
    distances_within_epsilon = np.less(distances, epsilon)
    neighbors = np.arange(len(data))[distances_within_epsilon]
    return neighbors


def dbscan(data: np.ndarray, epsilon: float, min_points: int) -> Dict:
    """
    DBSCAN algorithm. Pseudo code found at
    https://dl.acm.org/doi/pdf/10.1145/3068335.

    :param data: Array of points.
    :param min_points: Minimum # of points to determine core point.
    :param epsilon: Distance of circle to determine core point.
    :return: Dictionary with # of clusters and labels of points.
    """
    labels: List[Union[None, int]] = [None] * len(data)
    cluster_label = -1

    # foreach point p in database DB do
    for i in range(len(data)):
        # if label(p) != undefined then continue
        if labels[i] is not None:
            continue

        # Neighbors N = RangeQuery(DB, dist, p, e)
        neighbors = get_neighbors(data, i, epsilon)

        # if |N| < minPts then
        if len(neighbors) < min_points:
            # label(p) = Noise
            labels[i] = -1
            # continue
            continue

        # c = next cluster label
        cluster_label += 1
        # label(p) = c
        labels[i] = cluster_label

        # Seed set S = N \ {p}
        seed_list = list(neighbors)
        seed_set = set(neighbors)

        # foreach q in S do
        for j in seed_list:
            if i == j:
                continue

            # if label(q) = Noise then label(q) = c
            if labels[j] == -1:
                labels[j] = cluster_label

            # if label(q) != undefined then continue
            if labels[j] is not None:
                continue

            # Neighbors N = RangeQuery(DB, dist, q, e)
            neighbors = get_neighbors(data, j, epsilon)
            # label(q) = c
            labels[j] = cluster_label
            # if |N| < minPts then continue
            if len(neighbors) < min_points:
                continue

            # S = S union N
            for k in neighbors:
                if k not in seed_set:
                    seed_list.append(k)

            seed_set = seed_set.union(set(neighbors))

    return {
        'number_of_clusters': cluster_label + 1,
        'labels': np.array(labels)
    }
