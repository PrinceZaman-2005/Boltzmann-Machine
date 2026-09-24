import numpy as np
import matplotlib.pyplot as plt

from core.boltzmann_machine import BoltzmannMachine
from training.dataset import create_dataset
from training.training import Trainer


def calculate_test_accuracy(model, test_data):
    """
    Measure pixel-level reconstruction accuracy
    across the complete test set.
    """

    correct_pixels = 0
    total_pixels = 0

    for image in test_data:

        reconstruction = model.sample(
            image
        )

        correct_pixels += np.sum(
            image == reconstruction
        )

        total_pixels += image.size

    return correct_pixels / total_pixels


def main():
    # Create image dataset.
    train_data, test_data = create_dataset(
        num_samples=1000,
        image_size=8,
        test_ratio=0.2
    )

    # Create Boltzmann Machine.
    model = BoltzmannMachine(
        num_units=64,
        learning_rate=0.1
    )

    # Create trainer.
    trainer = Trainer(
        model=model,
        learning_rate=0.1,
        epochs=100
    )

    print("Training Boltzmann Machine")
    print("=" * 35)

    trainer.train(train_data)

    # Evaluate reconstruction on test set.
    test_accuracy = calculate_test_accuracy(
        model,
        test_data
    )

    print(
        f"\nTest Reconstruction Accuracy: "
        f"{test_accuracy:.2%}"
    )

    # Select a test image for visualization.
    original = test_data[0]

    reconstruction = model.sample(
        original
    )

    # Calculate reconstruction error.
    error = np.mean(
        (original - reconstruction) ** 2
    )

    print(
        f"Reconstruction Error: {error:.6f}"
    )

    # Convert vectors back to images.
    original_image = original.reshape(8, 8)
    reconstruction_image = reconstruction.reshape(8, 8)

    # Create visualization.
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(
        original_image,
        cmap="gray"
    )
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(
        reconstruction_image,
        cmap="gray"
    )
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()

    # Save instead of opening a GUI window.
    plt.savefig(
        "reconstruction.png",
        dpi=150
    )

    plt.close()

    print(
        "\nSaved visualization: "
        "reconstruction.png"
    )


if __name__ == "__main__":
    main()
