from typing import Dict, Union, Tuple, List, Any

import numpy as np

from project.dbscan import distance
from project.priority_queue import PriorityQueue


class Optics:
    """
    Class to create an object for runnning OPTICS.
    Pseudo code from https://en.wikipedia.org/wiki/OPTICS_algorithm.
    """

    def __init__(self, data: np.ndarray, epsilon=float('inf'), min_points=5):
        """
        Initialize an instance of OPTICS.

        :param data: Data to run OPTICS on.
        :param epsilon: Epsilon tunable parameter.
        :param min_points: Min points tunable parameter.
        """
        np.random.seed(1)
        np.random.shuffle(data)
        self.data = data
        self.epsilon = epsilon
        self.min_points = min_points

        self.ordered_list = []
        self.reachability_distances = {}
        self.visited = np.zeros(len(data), dtype=bool)

        self.core_distances = np.array(
            [self.core_distance(i) for i in range(len(data))])

    def get_neighbors(self, index: int):
        """
        Get indexes of points less than epsilon distance away from the
        point at index index.

        :param index: Index of data point in data.
        :return: Neighbors and distances to neighbors.
        """
        distances = np.sqrt(np.sum((self.data - self.data[index]) ** 2, axis=1))
        distances_within_epsilon = np.less(distances, self.epsilon)
        neighbors = np.arange(len(self.data))[distances_within_epsilon]
        return neighbors, distances[neighbors]

    def core_distance(self, index: int) -> Union[float, None]:
        """
        Get the core distance of a point.

        :param index: Index of data point in data.
        :return: Core distance of given point of None if not a core point.
        """
        neighbors, distances = self.get_neighbors(index)

        if len(neighbors) < self.min_points:
            return None

        return np.sort(distances)[self.min_points - 1]

    # function update(N, p, Seeds, eps, MinPts) is
    def _update(self, neighbors: np.ndarray, index: int,
                seeds: PriorityQueue) -> None:
        """
        Update the seeds priority queue.

        :param neighbors: Neighbors of data point.
        :param index: Index of data point to update.
        :param seeds: Queue of seeds.
        :return: None.
        """

        # coredist = core-distance(p, eps, MinPts)
        core_dist = self.core_distances[index]

        # for each o in N
        for neighbor in neighbors:

            # if o is not processed then
            if not self.visited[neighbor]:

                # new-reach-dist = max(coredist, dist(p,o))
                new_reachability_distance = max(core_dist,
                                                distance(self.data[index],
                                                         self.data[neighbor]))

                # if o.reachability-distance == UNDEFINED then
                if neighbor not in self.reachability_distances:

                    # o.reachability-distance = new-reach-dist
                    self.reachability_distances[
                        neighbor] = new_reachability_distance

                    # Seeds.insert(o, new-reach-dist)
                    seeds += (new_reachability_distance, neighbor)

                # else // o in Seeds, check for improvement
                else:

                    old_reachability_distance = self.reachability_distances[
                        neighbor]

                    # if new-reach-dist < o.reachability-distance then
                    if new_reachability_distance < old_reachability_distance:

                        # o.reachability-distance = new-reach-dist
                        self.reachability_distances[
                            neighbor] = new_reachability_distance

                        # Seeds.move-up(o, new-reach-dist)
                        seeds.remove((old_reachability_distance, neighbor))
                        seeds += (new_reachability_distance, neighbor)

    def run(self) -> Dict[int, float]:
        """
        Run the OPTICS algorithm.

        :return: Reachabilities and ordered points from algorithm.
        """

        # for each unprocessed point p of DB do
        for i in range(len(self.data)):
            if self.visited[i]:
                continue

            # N = getNeighbors(p, eps)
            neighbors, *_ = self.get_neighbors(i)

            # mark p as processed
            self.visited[i] = True

            # output p to the ordered list
            self.ordered_list.append(i)

            # if core-distance(p, eps, MinPts) != UNDEFINED then
            if self.core_distances[i] is not None:

                # Seeds = empty priority queue
                seeds = PriorityQueue()

                # update(N, p, Seeds, eps, MinPts)
                self._update(neighbors, i, seeds)

                # for each next q in Seeds do
                while len(seeds):
                    reachability, q = seeds.pop()

                    # N' = getNeighbors(q, eps)
                    q_neighbors, *_ = self.get_neighbors(q)

                    # mark q as processed
                    self.visited[q] = True

                    # output q to the ordered list
                    self.ordered_list.append(q)

                    # if core-distance(q, eps, MinPts) != UNDEFINED do
                    if self.core_distances[q] is not None:

                        # update(N', q, Seeds, eps, MinPts)
                        self._update(q_neighbors, q, seeds)

        reachability_dict = {}
        for point in self.ordered_list:
            reachability_dict[point] = self.reachability_distances.get(point, 0)

        return reachability_dict


def clusterize_optics(reachabilities: Dict[int, float], threshold: float)\
        -> Dict[str, Union[int, Dict[int, int]]]:
    """
    Make clusters from the results of running OPTICS.

    :param reachabilities: Ordered points and their reachabilites.
    :param threshold: Threshold to cluster at.
    :return: Cluster labels and number of clusters found.
    """
    labels = {}

    current_label = 0
    incremented_last = False
    reachability_items = list(reachabilities.items())
    for i, reach in enumerate(reachability_items):
        point, reachability = reach
        if reachability > threshold:
            labels[point] = -1

            if not incremented_last:
                current_label += 1
                incremented_last = True

            if i < len(reachability_items) - 1 and reachability_items[
                i + 1][1] < threshold:
                labels[point] = current_label
        else:
            incremented_last = False
            labels[point] = current_label

    label_values = np.fromiter(labels.values(), dtype=int)

    has_noise = 1 if -1 in label_values else 0

    return {
        'number_of_clusters': len(np.unique(label_values)) - has_noise,
        'labels': labels,
    }


def main():
    pass


if __name__ == '__main__':
    main()
