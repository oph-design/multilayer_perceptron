import numpy as np
from .mathematics import sigmoid, sigmoid_prime


class Layer:

    def __init__(self, prev_count: int, count: int, size: int):
        """initializes  weights and biases"""
        self.biases = np.random.rand(count)
        self.weights = np.random.rand(count, prev_count)
        self.weighted_sums = np.zeros(count)
        self.errors = np.zeros((size, count))
        self.bias_delta = np.zeros(self.biases.shape)
        self.weight_delta = np.zeros(self.weights.shape)

    def get_neurons(self, prev_neurons: np.ndarray) -> np.ndarray:
        """returns the activation for all neurons"""
        self.weighted_sums = np.dot(self.weights, prev_neurons) + self.biases
        return sigmoid(self.weighted_sums)

    def calculate_error(
        self, post_weights: np.ndarray, post_error: np.ndarray, index: int
    ) -> np.ndarray:
        """calculate the error of each neuron of the layer"""
        gradient = sigmoid_prime(self.weighted_sums)
        self.errors[index] = np.dot(post_weights.T, post_error) * gradient
        return self.errors[index]

    def learn(self, lr: float, prev_activation: np.ndarray, index: int) -> None:
        """adjust the parameters based on the current error"""
        change_rate = lr * np.outer(self.errors[index], prev_activation)
        self.weight_delta = self.weight_delta + change_rate
        self.bias_delta = self.bias_delta + lr * self.errors[index]

    def apply_changes(self, size: int) -> None:
        """means the adjustments and applys them to the weights"""
        self.biases = self.biases - self.bias_delta / size
        self.weights = self.weights - self.weight_delta / size
        self.bias_delta = np.zeros(self.biases.shape)
        self.weight_delta = np.zeros(self.weights.shape)
