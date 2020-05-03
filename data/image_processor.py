from pathlib import Path
from typing import Tuple

import numpy as np

import cv2


def process_image(filename, resize: Tuple[int, int] = None):
    relative_dir = Path(__file__).parent
    image_path = relative_dir / filename

    image = cv2.imread(str(image_path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if resize:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

    np.random.seed(1)

    binary: np.ndarray = cv2.threshold(image, 128, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    binary[binary == 0] = 1
    binary[binary == 255] = 0

    return binary


def main():
    binary = process_image('images/spiral.png')
    print(binary)
    print(binary.shape)


if __name__ == '__main__':
    main()
