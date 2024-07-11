import numpy as np
from .mathematics import sigmoid, sigmoid_prime, soft_max


class Layer:
    def __init__(self, w: np.ndarray, b: np.ndarray, size: int, activation: int):
        """initializes  weights and biases"""
        self.biases = b
        self.weights = w
        self.activations = np.zeros((size, len(b)))
        self.errors = np.zeros((size, len(b)))
        self.bias_delta = np.zeros(b.shape)
        self.weight_delta = np.zeros(w.shape)
        self.activation = sigmoid if activation != 0 else soft_max

    def get_neurons(self, prev_neurons: np.ndarray, index: int) -> np.ndarray:
        """returns the activation for all neurons"""
        weighted_sum = np.dot(self.weights, prev_neurons) + self.biases
        self.activations[index] = sigmoid(weighted_sum)
        return self.activations[index]

    def calculate_error(
        self, post_weights: np.ndarray, post_error: np.ndarray, index: int
    ) -> np.ndarray:
        """calculate the error of each neuron of the layer"""
        gradient = self.activations[index] * (1.0 - self.activations[index])
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
