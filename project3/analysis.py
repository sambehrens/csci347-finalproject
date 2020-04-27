from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import get_worms_data, get_barcode, get_karl, get_pig, get_world
from project3.dbscan import dbscan

plot_size = 0.25


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


def plot_worms():
    plot(get_worms_data(), 'worms')


def plot_barcode():
    plot(get_barcode(), 'barcode')


def plot_by_type(data: np.ndarray, classes: np.ndarray,
                 class_names: Dict[int, str]):
    """
    Plots data set according to labels.

    :return: None.
    """
    for i, label in enumerate(np.unique(classes)):
        plt.scatter(
            data[classes == label, 0],
            data[classes == label, 1],
            s=plot_size, label=class_names[int(label)])

    plt.title('Scatter plot by class')
    plt.legend(loc='lower right')
    plt.show()


def plot_dbscan_clusters(data, epsilon, min_points):
    """
    Plots the DBSCAN clusters.

    :param data: Data to run DBSCAN on.
    :param epsilon: Epsilon parameter.
    :param min_points: Min points parameter.
    :return: None.
    """
    results = dbscan(data, epsilon, min_points)

    labels = {-1: 'Noise'}

    for i, label in enumerate(np.unique(results['labels'])):
        plt.scatter(
            data[results['labels'] == label, 0],
            data[results['labels'] == label, 1],
            s=plot_size, label=labels.get(label, f'Cluster {label}'))

    plt.title(f'Clusters found with DBSCAN '
              f'(epsilon={epsilon}, min_points={min_points})')
    # plt.legend(loc='lower right')
    plt.show()


def get_dbscan_results(data, epsilons, min_points):
    """
    Gets a matrix of cluster counts from running DBScan.

    :param data: Data to run DBScan on.
    :param epsilons: List of epsilon values to use.
    :param min_points: List of min point values to use.
    :return: Results of running DBScan.
    """
    results = []

    for epsilon in epsilons:
        epsilon_results = []
        for min_point in min_points:
            epsilon_results.append(
                dbscan(data, epsilon, min_point)['number_of_clusters'])
        results.append(epsilon_results)

    return results


def analyze_dbscan():
    """
    Performs analysis on the results of DBScan.

    :return: None.
    """
    epsilons = [x / 1000 for x in range(12, 22, 2)]
    min_points = range(3, 8)
    db_scan_results = get_dbscan_results(get_worms_data(),
                                         epsilons, min_points)

    print_markdown_rows('ε', epsilons, 'mp', min_points, db_scan_results)

    dbscan_column = np.array(db_scan_results).flatten()

    epsilons_column = [[e] * 5 for e in epsilons]
    epsilons_column = np.array(epsilons_column).flatten()

    min_points_column = list(min_points) * 5

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(min_points_column, epsilons_column, dbscan_column,
                    linewidth=0.2,
                    antialiased=True,
                    color='lightblue')

    _ = Axes3D

    ax.set_xlabel('min_points')
    ax.set_ylabel('epsilon')
    ax.set_zlabel('# of Clusters')
    ax.set_title('DBSCAN Cluster Count vs Parameters for Original Data')
    plt.show()


def main():
    # plot_worms()
    # plot_barcode()
    # plot_by_type()
    # data = get_barcode(n_samples=20_000)
    # data = get_karl()
    data = get_world()
    plot_dbscan_clusters(data, 2.2, 5)
    pass


if __name__ == '__main__':
    main()
