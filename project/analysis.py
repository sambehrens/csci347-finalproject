import timeit
from typing import Dict, Union

import numpy as np
from matplotlib import pyplot as plt

from data import get_data_from_image
from project.dbscan import dbscan
from project.optics import clusterize_optics, Optics

plot_size = 20


def print_markdown_rows(row_title, row_labels, column_title, column_labels,
                        data):
    """
    Print a table formatted for markdown.

    :param row_title: Title of the rows.
    :param row_labels: Labels for the rows.
    :param column_title: Title of the columns.
    :param column_labels: Labels for the columns.
    :param data: Data to put in the table.
    :return: None.
    """
    header = (
        f'| | '
        f'{" | ".join(map(lambda x: f"{column_title}={x}", column_labels))} |')
    divider = f'| --- | {" | ".join(map(lambda x: "---", column_labels))} |'

    rows = [header, divider]
    for i, row in enumerate(data):
        row_markup = " | ".join(map(str, row))
        rows.append(
            f'| {f"**{row_title}={row_labels[i]}**"} | {row_markup} |')

    print('\n'.join(rows))


def plot(data, name):
    """
    Plots a data set.

    :return: None
    """
    plt.scatter(data[:, 0], data[:, 1], s=plot_size)

    plt.title(f'Scatter plot of {name} data set')
    plt.show()


def plot_clusters(data, results: Dict[str, Union[int, np.ndarray]],
                  epsilon: float, min_points: int, name: str, **kwargs):
    """
    Plots clusters.

    :param data:
    :param name:
    :param results:
    :param epsilon: Epsilon parameter.
    :param min_points: Min points parameter.
    :return: None.
    """
    labels = {-1: 'Noise'}

    for i, label in enumerate(np.unique(results['labels'])):
        plt.scatter(
            data[results['labels'] == label, 0],
            data[results['labels'] == label, 1],
            s=plot_size, label=labels.get(label, None), marker='o')

    if -1 in np.unique(results['labels']):
        plt.legend(loc='lower right')

    title = (
        f'{results["number_of_clusters"]} Clusters found '
        f'with {name}\n'
        f'(epsilon={epsilon}, min_points={min_points}, n_samples='
        f'{len(data)}')

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
        'resize': (80, 60)
    }
    alg_args = {
        'epsilon': 20,
        'min_points': 3,
        'threshold': 1.5,
    }
    data = get_data_from_image('../data/images/barcode.jpg', **data_args)
    # plot_dbscan(data, 3.5, 2)
    plot_optics(data, **alg_args)

    # result_labels = OPTICS(max_eps=20, min_samples=5).fit_predict(data)
    #
    # labels = {-1: 'Noise'}
    #
    # for i, label in enumerate(np.unique(result_labels)):
    #     plt.scatter(
    #         data[result_labels == label, 0],
    #         data[result_labels == label, 1],
    #         s=plot_size, label=labels.get(label, None))
    #
    # plt.show()
    pass


if __name__ == '__main__':
    main()
