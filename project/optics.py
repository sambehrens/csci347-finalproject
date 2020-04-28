from heapq import heappush, heapify, heappop
from typing import Dict, Union, Tuple, List

import numpy as np

from data import get_world
from project.dbscan import distance


class Point:
    def __init__(self, index: int, coords: np.ndarray):
        self.coords = coords
        self.index = index
        self.reachability = None
        self.visited = False


def get_neighbors(data: np.ndarray, points: List[Point], point: Point, epsilon):
    """
    Gets indexes of points less than epsilon distance away from the
    point at index index.

    :param data: Array of points.
    :param points:
    :param point:
    :param epsilon: Distance away from the point.
    :return: Indexes of the neighbors.
    """
    distances = np.sqrt(np.sum((data - point.coords) ** 2, axis=1))
    distances_within_epsilon = np.less(distances, epsilon)
    neighbors = np.arange(len(data))[distances_within_epsilon]
    neighbor_points = list(np.array(points)[neighbors])
    return neighbor_points, distances[neighbors]


def core_distance(data: np.ndarray, points: List[Point], point: Point,
                  epsilon: float, min_points: int) -> Union[float, None]:
    """

    :param data:
    :param points:
    :param point:
    :param epsilon:
    :param min_points:
    :return:
    """
    neighbors, distances = get_neighbors(data, points, point, epsilon)

    if len(neighbors) < min_points:
        return None

    return np.sort(distances)[min_points - 1]


# function update(N, p, Seeds, eps, MinPts) is
def update(neighbors: List[Point], data: np.ndarray, points: List[Point], point,
           seeds: list, epsilon: float, min_points: int) -> None:
    # coredist = core-distance(p, eps, MinPts)
    core_dist = core_distance(data, points, point, epsilon, min_points)

    # for each o in N
    for neighbor in neighbors:

        # if o is not processed then
        if not neighbor.visited:

            # new-reach-dist = max(coredist, dist(p,o))
            new_reachability_distance = max(core_dist,
                                            distance(point.coords,
                                                     neighbor.coords))

            # if o.reachability-distance == UNDEFINED then
            if neighbor.reachability is None:

                # o.reachability-distance = new-reach-dist
                neighbor.reachability = new_reachability_distance

                # Seeds.insert(o, new-reach-dist)
                heappush(seeds,
                         (new_reachability_distance, neighbor.index, neighbor))

            # else // o in Seeds, check for improvement
            else:

                # if new-reach-dist < o.reachability-distance then
                if new_reachability_distance < neighbor.reachability:

                    # o.reachability-distance = new-reach-dist
                    neighbor.reachability = new_reachability_distance

                    # Seeds.move-up(o, new-reach-dist)
                    seeds = list(filter(lambda x: x[1] != neighbor, seeds))
                    heapify(seeds)
                    heappush(seeds, (
                        new_reachability_distance, neighbor.index, neighbor))


def optics(data: np.ndarray, epsilon: float = float('inf'), min_points: int =
5) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param data:
    :param epsilon:
    :param min_points:
    :return:
    """
    # reachability_distances = np.zeros(len(data))
    # visited = np.zeros(len(data), dtype=bool)

    points: List[Point] = [Point(i, coords) for i, coords in enumerate(data)]

    ordered_list = []

    # for each unprocessed point p of DB do
    for point in points:
        if point.visited:
            continue

        # N = getNeighbors(p, eps)
        neighbors, *_ = get_neighbors(data, points, point, epsilon)

        # mark p as processed
        point.visited = True

        # output p to the ordered list
        ordered_list.append(point)

        # if core-distance(p, eps, MinPts) != UNDEFINED then
        if core_distance(data, points, point, epsilon, min_points) is not None:

            # Seeds = empty priority queue
            seeds = []

            # update(N, p, Seeds, eps, MinPts)
            update(neighbors, data, points, point, seeds, epsilon, min_points)

            # for each next q in Seeds do
            while len(seeds):
                reachability, _, q = heappop(seeds)

                # N' = getNeighbors(q, eps)
                q_neighbors, *_ = get_neighbors(data, points, q, epsilon)

                # mark q as processed
                q.visited = True

                # output q to the ordered list
                ordered_list.append(q)

                # if core-distance(q, eps, MinPts) != UNDEFINED do
                if core_distance(data, points, q, epsilon,
                                 min_points) is not None:

                    # update(N', q, Seeds, eps, MinPts)
                    update(q_neighbors, data, points, q, seeds, epsilon,
                           min_points)

    reachabilities = np.fromiter(
        map(lambda x: x.reachability if x.reachability else 0, ordered_list),
        dtype=float)
    points_indices = np.fromiter(map(lambda x: x.index, ordered_list),
                                 dtype=int)

    return reachabilities, points_indices


def clusterize_optics(reachabilities, points, threshold):
    labels = np.empty(len(points))

    current_label = 0
    for i in range(len(reachabilities)):
        if reachabilities[i] > threshold:
            labels[points[i]] = -1
            current_label += 1
        else:
            labels[points[i]] = current_label

    return {
        'number_of_clusters': len(np.unique(labels)) - 1,
        'labels': labels,
    }


def main():
    data = get_world(n_samples=100)
    reachabilities, points = optics(data, float('inf'), 5)
    print(reachabilities)
    print(points)


if __name__ == '__main__':
    main()
