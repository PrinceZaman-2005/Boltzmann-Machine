import numpy as np


def calculate_energy(states, weights, bias):
    """
    Calculate the energy of a Boltzmann Machine state.

    E = -1/2 * sWs - b·s

    Lower energy means the state is more favorable
    according to the model.
    """

    states = np.asarray(states)

    interaction_energy = (
        -0.5
        * states
        @ weights
        @ states
    )

    bias_energy = -np.dot(
        bias,
        states
    )

    return interaction_energy + bias_energy
