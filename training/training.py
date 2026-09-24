import numpy as np


class Trainer:
    """
    Handles training for the Boltzmann Machine.

    Training objective:
        Learn to reconstruct the structure
        of the training images.

    The Trainer is responsible for:
    - running epochs
    - processing training images
    - collecting reconstruction error
    - tracking training history

    Contrastive Divergence handles the
    actual parameter updates.
    """

    def __init__(
        self,
        model,
        learning_rate=0.1,
        epochs=100
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.history = {
            "reconstruction_error": []
        }

    def reconstruction_error(
        self,
        original,
        reconstruction
    ):
        """
        Calculate the difference between
        the original and reconstructed image.
        """

        return np.mean(
            (original - reconstruction) ** 2
        )

    def train(self, data):
        """
        Train the Boltzmann Machine.

        Each image is used as a visible state.
        """

        data = np.asarray(data)

        for epoch in range(1, self.epochs + 1):

            errors = []

            for image in data:

                result = self.model.train_step(
                    image
                )

                reconstruction = result[
                    "reconstructed"
                ]

                error = self.reconstruction_error(
                    image,
                    reconstruction
                )

                errors.append(error)

            mean_error = np.mean(errors)

            self.history[
                "reconstruction_error"
            ].append(mean_error)

            if (
                epoch == 1
                or epoch % 10 == 0
                or epoch == self.epochs
            ):
                print(
                    f"Epoch {epoch}/{self.epochs} "
                    f"| Reconstruction Error: "
                    f"{mean_error:.6f}"
                )

        return self.history
