import numpy as np

from core.energy import calculate_energy
from core.contrastive_divergence import ContrastiveDivergence


class BoltzmannMachine:
    """
    A simple Boltzmann Machine built from scratch.

    Responsibilities:
    - store model parameters
    - calculate neuron probabilities
    - sample stochastic states
    - train using Contrastive Divergence
    """

    def __init__(
        self,
        num_units,
        learning_rate=0.1
    ):
        self.num_units = num_units
        self.learning_rate = learning_rate

        # Connection weights.
        self.weights = (
            np.random.randn(
                num_units,
                num_units
            ) * 0.1
        )

        # No self-connections.
        np.fill_diagonal(
            self.weights,
            0
        )

        # Neuron biases.
        self.bias = np.zeros(
            num_units
        )

        # Learning algorithm.
        self.cd = ContrastiveDivergence(
            learning_rate=learning_rate
        )

    def probability(self, states):
        """
        Calculate activation probabilities
        for all neurons.
        """

        states = np.asarray(states)

        input_signal = (
            np.dot(
                states,
                self.weights
            )
            + self.bias
        )

        return 1 / (
            1 + np.exp(-input_signal)
        )

    def sample(self, states):
        """
        Sample a new stochastic state.
        """

        probabilities = self.probability(
            states
        )

        return (
            np.random.random(
                self.num_units
            ) < probabilities
        ).astype(int)

    def energy(self, states):
        """
        Calculate the energy of a state.
        """

        return calculate_energy(
            states,
            self.weights,
            self.bias
        )

    def train_step(self, states):
        """
        Perform one Contrastive Divergence step.
        """

        result = self.cd.train_step(
            states,
            self.weights,
            self.bias
        )

        return result
