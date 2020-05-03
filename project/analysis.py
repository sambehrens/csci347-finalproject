from typing import Dict, Union

import numpy as np
from matplotlib import pyplot as plt

from data import get_data_from_image, get_2d_data_from_file
from project.dbscan import dbscan
from project.optics import clusterize_optics, Optics

plot_size = 20


def plot_clusters(data, results: Dict[str, Union[int, np.ndarray]],
                  epsilon: float, min_points: int, name: str, **kwargs):
    labels = {-1: 'Noise'}

    for i, label in enumerate(np.unique(results['labels'])):
        plt.scatter(
            data[results['labels'] == label, 0],
            data[results['labels'] == label, 1],
            s=plot_size, label=labels.get(label, None), marker='o')

    if -1 in np.unique(results['labels']):
        plt.legend(loc='lower right')

    title = (
        f'{results["number_of_clusters"]} Clusters found with {name}\n'
        f'(epsilon={epsilon}, min_points={min_points}, n_samples={len(data)}')

    for arg, value in kwargs.items():
        title += f', {arg}={value}'

    title += ')'

    plt.title(title)
    plt.show()


def plot_dbscan(data: np.ndarray, epsilon, min_points):
    results = dbscan(data, epsilon, min_points)
    plot_clusters(data, results, epsilon, min_points, 'DBSCAN')


def plot_optics(data, epsilon, min_points, threshold):
    reachabilities = Optics(data, epsilon, min_points).run()
    results = clusterize_optics(reachabilities, threshold)

    colors_dict = {-1: 'gray'}
    color_options = ['g', 'y', 'darkviolet', 'salmon', 'darkorange']
    color_swapper = 0
    for point, label in results['labels'].items():
        if label == -1:
            continue
        colors_dict[label] = color_options[color_swapper % len(color_options)]
        color_swapper += 1

    colors = [colors_dict[label] for label in results['labels'].values()]
    plt.hlines(threshold, 0, len(results['labels']), linestyles='dashed',
               colors='r', label='threshold')
    plt.bar(np.arange(len(reachabilities)), reachabilities.values(),
            color=colors, snap=False)
    plt.title('Reachabilities of ordered points')
    plt.ylabel('Reachability')
    plt.show()

    label_list = np.array(
        sorted(results['labels'].items(), key=lambda x: x[0]))[:, 1]
    results['labels'] = label_list
    plot_clusters(data, results, epsilon, min_points, 'OPTICS',
                  threshold=threshold)


def main():
    data_args = {
        'n_samples': None,
        'resize': (40, 30)
    }
    alg_args = {
        'epsilon': float('inf'),
        'min_points': 2,
        'threshold': 2.5,
    }
    # data = get_2d_data_from_file('../data/toy.txt')
    data = get_data_from_image('../data/images/spiral.png', **data_args)
    # plot_dbscan(data, 2.5, 3)
    plot_optics(data, **alg_args)


if __name__ == '__main__':
    main()
