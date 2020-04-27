import unittest

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris

from project3.dbscan import get_neighbors, distance, dbscan


class TestDbscan(unittest.TestCase):
    def test_get_neighbors(self):
        data = np.arange(6).reshape(3, 2)
        self.assertTrue(np.array_equal([0, 1], get_neighbors(data, 0, 3)))

    def test_distance(self):
        self.assertEqual(5, distance(np.array([1, 2]), np.array([4, 6])))

    def test_dbscan(self):
        data = '''\
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
'''.splitlines()
        data = np.array([[int(y) for y in x.split()] for x in data])
        np.random.seed(1)
        np.random.shuffle(data)
        min_points = 2
        epsilon = 3
        result = dbscan(data, epsilon, min_points)
        result['labels'] = list(result['labels'])
        expected = {
            'number_of_clusters': 2,
            'labels': [0, 0, 0, 1, 0, 0, 0, -1, 1, 1, 1, 0]
        }

        self.assertEqual(expected, result)

    def test_dbscan_iris(self):
        iris_data = load_iris()
        data = iris_data['data']
        eps = 0.4
        min_pts = 5
        dbs = DBSCAN(eps=eps, min_samples=min_pts)
        dbs.fit(data)
        self.assertEqual(list(dbs.labels_),
                         list(dbscan(data, eps, min_pts)['labels']))


if __name__ == '__main__':
    unittest.main()
