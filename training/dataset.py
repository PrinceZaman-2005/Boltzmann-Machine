import numpy as np


def create_dataset(
    num_samples=100,
    image_size=8,
    test_ratio=0.2
):
    """
    Create a simple binary image dataset.

    The images contain simple horizontal,
    vertical, and diagonal patterns.

    Returns:
        train_data
        test_data
    """

    images = []

    for _ in range(num_samples):

        image = np.zeros(
            (image_size, image_size),
            dtype=int
        )

        pattern = np.random.randint(0, 3)

        if pattern == 0:
            # Horizontal line
            row = np.random.randint(
                0,
                image_size
            )
            image[row, :] = 1

        elif pattern == 1:
            # Vertical line
            column = np.random.randint(
                0,
                image_size
            )
            image[:, column] = 1

        else:
            # Main diagonal
            np.fill_diagonal(
                image,
                1
            )

        images.append(
            image.flatten()
        )

    data = np.array(images)

    test_size = int(
        num_samples * test_ratio
    )

    test_data = data[:test_size]
    train_data = data[test_size:]

    return train_data, test_data
