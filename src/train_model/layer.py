import numpy as np
from .functions import sigmoid


class Layer:

    def __init__(self, prev_count: int, count: int):
        """initializes  weights and biases"""
        self.biases = np.random.rand(count)
        self.weights = np.random.rand(count, prev_count)

    def get_neurons(self, prev_neurons: np.ndarray) -> np.ndarray:
        """returns the activation for all neurons"""
        return sigmoid(np.dot(self.weights, prev_neurons) + self.biases)
