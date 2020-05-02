from pathlib import Path

import numpy as np
from sklearn.utils import resample

from data.image_processor import process_image


def get_2d_data_from_file(filename: str) -> np.ndarray:
    """
    Convert text file to data set.

    :param filename: Name of file to convert.
    :return: Data set.
    """
    relative_dir = Path(__file__).parent
    data_path = relative_dir / filename

    with open(data_path) as file:
        data = np.loadtxt(file)

    return data


def get_data_from_image(filename: str, resize=None,
                        n_samples=None) -> np.ndarray:
    """
    Get data set from image.

    :param filename: Name of image file.
    :param resize: Dimensions to resize image to.
    :param n_samples: Number of samples to re-sample to.
    :return: Data set.
    """
    image = process_image(filename, resize)

    x, y = np.nonzero(image)
    max_x = np.max(x)

    coords = np.stack((y, max_x - x), axis=1)

    if n_samples is not None:
        return resample(coords, n_samples=n_samples, random_state=1)

    return coords


if __name__ == '__main__':
    data = get_data_from_image('images/spiral.png')
    print(data)
    print(data.shape)
