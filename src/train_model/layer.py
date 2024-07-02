import numpy as np
from .mathematics import sigmoid, sigmoid_prime


class Layer:

    def __init__(self, prev_count: int, count: int):
        """initializes  weights and biases"""
        self.biases = np.random.rand(count)
        self.weights = np.random.rand(count, prev_count)
        self.errors = np.zeros(count)
        self.weighted_sums = np.zeros(count)

    def get_neurons(self, prev_neurons: np.ndarray) -> np.ndarray:
        """returns the activation for all neurons"""
        self.weighted_sums = np.dot(self.weights, prev_neurons) + self.biases
        return sigmoid(self.weighted_sums)

    def get_weights(self) -> np.ndarray:
        """return the current weights of the layer"""
        return self.weights

    def calculate_error(
        self, post_weights: np.ndarray, post_error: np.ndarray
    ) -> np.ndarray:
        """calculate the error of each neuron of the layer"""
        self.errors = (post_weights.T * post_error) * sigmoid_prime(self.weighted_sums)
        return self.errors

    def apply_error(self, lr: float) -> None:
        """adjust the parameters based on the current error"""
        self.biases = self.biases - lr * self.errors
        self.weights = self.weights - lr * self.errors
