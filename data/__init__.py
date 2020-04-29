from pathlib import Path

import numpy as np
from sklearn.utils import resample

from data.image_processor import process_image


def get_2d_data_from_file(filename) -> np.ndarray:
    """

    :param filename:
    :return:
    """
    relative_dir = Path(__file__).parent
    data_path = relative_dir / filename

    with open(data_path) as file:
        data = np.loadtxt(file)

    return data


def get_worms_data(n_samples=10_000) -> np.ndarray:
    """
    Gets the worms data set as a numpy array.

    :param n_samples: Number of samples to re-sample.
    :return: The numpy array.
    """
    data = get_2d_data_from_file('worms.txt')

    return resample(data, n_samples=n_samples, random_state=1)


def get_data_from_binary_image(filename: str, resize=None) -> np.ndarray:
    """

    :param filename:
    :param resize:
    :return:
    """
    image = process_image(filename, resize)

    x, y = np.nonzero(image)
    max_x = np.max(x)

    coords = np.stack((y, max_x - x), axis=1)

    return coords


def get_data_from_image(filename, n_samples=None, resize=(80, 60)):
    data = get_data_from_binary_image(filename, resize=resize)

    if n_samples is not None:
        return resample(data, n_samples=n_samples, random_state=1)

    return data

def get_barcode(**kwargs):
    return get_data_from_image('images/barcode.jpg', **kwargs)


def get_karl(**kwargs):
    return get_data_from_image('images/karl.jpg', **kwargs)


def get_pig(**kwargs):
    return get_data_from_image('images/pig.jpg', **kwargs)


def get_world(**kwargs):
    return get_data_from_image('images/world.jpg', **kwargs)


def get_toy(**kwargs):
    return get_data_from_image('images/toy.png', **kwargs)


if __name__ == '__main__':
    # data = get_worms_data()
    data = get_barcode()
    print(data)
    print(data.shape)
