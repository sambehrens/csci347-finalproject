from heapq import heappush
from unittest import TestCase
import matplotlib.pyplot as plt

import numpy as np

from project.optics import core_distance, update, get_neighbors, optics, \
    clusterize_optics


class Test(TestCase):

    @staticmethod
    def get_data_from_string(string):
        lines = string.splitlines()
        return np.array([[int(y) for y in x.split()] for x in lines])

    def test_core_distance(self):
        data = self.get_data_from_string('''\
0 0
1 0
2 0
3 0
4 0
5 0
6 0
20 0
0 10
1 10
2 10
3 10
''')
        core_dist = core_distance(data, index=1, epsilon=10, min_points=2)
        self.assertEqual(1, core_dist)

    def test_update(self):
        data = self.get_data_from_string('''\
0 0
1 0
2 0
3 0
4 0
5 0
6 0
20 0
0 10
1 10
2 10
3 10
''')
        index = 1
        eps = 3
        neighbors, *_ = get_neighbors(data, index, eps)
        seeds = []
        heappush(seeds, (1, 1))

        visited = np.zeros(len(data), dtype=bool)
        visited_copy = visited.copy()

        reachability_distances = np.full(len(data), -1)
        reachability_distances_copy = reachability_distances.copy()

        update(neighbors, data, index, seeds, eps, 2, visited,
               reachability_distances)

        np.testing.assert_array_equal(reachability_distances_copy, reachability_distances)
        np.testing.assert_array_equal(visited_copy, visited)
        np.testing.assert_array_equal(seeds, [(1, 1)])

    def test_optics(self):
        data = self.get_data_from_string('''\
0 0
1 1
10 10
11 11
20 20
21 21
30 30
31 31
''')
        reachabilities, points = optics(data, epsilon=2, min_points=2)
        plt.bar(np.arange(len(reachabilities)), reachabilities)
        plt.show()

    def test_clusterize_optics(self):
        data = self.get_data_from_string('''\
0 0
1 1
20 20
21 21
''')
        reachabilities, points = optics(data, epsilon=100, min_points=2)
        results = clusterize_optics(reachabilities, points, threshold=2)

        np.testing.assert_array_equal(np.array([0, 0, -1, 1]), results['labels'])
        self.assertEqual(2, results['number_of_clusters'])


