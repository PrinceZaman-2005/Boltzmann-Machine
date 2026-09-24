import numpy as np

from core.energy import calculate_energy


class ContrastiveDivergence:
    """
    Contrastive Divergence learning.

    CD-1:
    1. Start from the real data.
    2. Calculate the positive phase.
    3. Sample a new state.
    4. Calculate the negative phase.
    5. Update weights and biases.
    """

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """
        Convert an input signal into a probability.
        """

        return 1 / (1 + np.exp(-x))

    def sample(self, states, weights, bias):
        """
        Sample a new binary state from the current state.
        """

        probabilities = self.sigmoid(
            np.dot(states, weights) + bias
        )

        return (
            np.random.random(
                len(states)
            ) < probabilities
        ).astype(int)

    def train_step(self, states, weights, bias):
        """
        Perform one CD-1 learning step.
        """

        states = np.asarray(
            states,
            dtype=int
        )

        # -------------------------
        # Positive phase
        # -------------------------

        positive = np.outer(
            states,
            states
        )

        # -------------------------
        # Negative phase
        # -------------------------

        reconstructed = self.sample(
            states,
            weights,
            bias
        )

        negative = np.outer(
            reconstructed,
            reconstructed
        )

        # -------------------------
        # Update parameters
        # -------------------------

        weight_update = (
            self.learning_rate
            * (positive - negative)
        )

        bias_update = (
            self.learning_rate
            * (states - reconstructed)
        )

        weights += weight_update
        bias += bias_update

        # Keep weights symmetric.
        weights[:] = (
            weights + weights.T
        ) / 2

        # Remove self-connections.
        np.fill_diagonal(
            weights,
            0
        )

        return {
            "reconstructed": reconstructed,
            "energy_before": calculate_energy(
                states,
                weights,
                bias
            ),
            "energy_after": calculate_energy(
                reconstructed,
                weights,
                bias
            )
        }
