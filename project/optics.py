from typing import Dict, Union, Tuple, List

import numpy as np

from data import get_world
from project.dbscan import distance
from project.priority_queue import PriorityQueue


class Optics:
    def __init__(self, data: np.ndarray, epsilon=float('inf'), min_points=5):
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

    def get_neighbors(self, index):
        """
        Gets indexes of points less than epsilon distance away from the
        point at index index.

        :param index:
        :return:
        """
        distances = np.sqrt(np.sum((self.data - self.data[index]) ** 2, axis=1))
        distances_within_epsilon = np.less(distances, self.epsilon)
        neighbors = np.arange(len(self.data))[distances_within_epsilon]
        return neighbors, distances[neighbors]

    def core_distance(self, index) -> Union[float, None]:
        """

        :param index:
        :return:
        """
        neighbors, distances = self.get_neighbors(index)

        if len(neighbors) < self.min_points:
            return None

        return np.sort(distances)[self.min_points - 1]

    # function update(N, p, Seeds, eps, MinPts) is
    def update(self, neighbors: np.ndarray, index,
               seeds: PriorityQueue) -> None:

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

        :return:
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
                self.update(neighbors, i, seeds)

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
                        self.update(q_neighbors, q, seeds)

        # reachabilities = np.zeros(len(self.data), dtype=float)
        # for index, distance in self.reachability_distances.items():
        #     reachabilities[index] = distance

        reachability_dict = {}
        for point in self.ordered_list:
            reachability_dict[point] = self.reachability_distances.get(point, 0)

        # print('number of zeros', len(reachabilities[reachabilities == 0]))
        #
        # point_indices = np.array(self.ordered_list)

        return reachability_dict


def clusterize_optics(reachabilities, threshold: float):
    labels = {}

    current_label = 0
    incremented_last = False
    for point, reachability in reachabilities.items():
        if reachability > threshold:
            labels[point] = -1

            if not incremented_last:
                current_label += 1
                incremented_last = True
        else:
            incremented_last = False
            labels[point] = current_label

    print(labels)

    return {
        'number_of_clusters': len(np.unique(labels)) - 1,
        'labels': labels,
    }


def main():
    data = get_world(n_samples=2000)
    reachabilities, points = Optics(data).run()
    print(reachabilities)
    print(points)


if __name__ == '__main__':
    main()
