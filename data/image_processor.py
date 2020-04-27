from pathlib import Path
import numpy as np

import cv2


def process_image(filename):
    relative_dir = Path(__file__).parent
    image_path = relative_dir / filename

    image = cv2.imread(str(image_path))

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    np.random.seed(1)

    binary: np.ndarray = cv2.threshold(grayscale, 128, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    binary[binary == 0] = 1
    binary[binary == 255] = 0

    return binary


def main():
    binary = process_image('images/barcode.jpg')
    print(binary)
    print(binary.shape)


if __name__ == '__main__':
    main()
