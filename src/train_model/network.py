import numpy as np
import pandas as pd
from .layer import Layer
from .mathematics import bce_prime, sigmoid_prime, sigmoid


class Network:

    def __init__(self, conf: dict, data: pd.DataFrame):
        """initializes layers of the neural Network based on conf"""
        self.data = data
        self.epochs = conf["epochs"]
        self.rate = conf["learning_rate"]
        self.size = conf["batch_size"]
        self.dim = np.insert([30, 2], 1, conf["layer"])
        self.layers = [
            Layer(self.dim[x - 1], self.dim[x], self.size)
            for x in range(1, len(self.dim))
        ]

    def fit(self):
        """performs gradient descent for all layers in epoch time"""
        for i in range(self.epochs):
            batch = self.data.loc[self.data["Batch"] == i % self.size].values
            target = batch[:, 1:2].flatten()
            input = batch[:, 2:]
            prediction = self.feed_forward(input)
            error = self.calc_out_layer_error(target, prediction)
            self.propagate_backwards(error)
            self.adjust_parameters(input)

    def feed_forward(self, batch: np.ndarray) -> np.ndarray:
        """calculates the prediction for the current weights"""
        res = []
        for data in batch:
            for layer in self.layers:
                data = layer.get_neurons(data)
            res.append(data)
        return np.array(res)

    def calc_out_layer_error(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """calculates the error in the output layer"""
        gradient = sigmoid_prime(self.layers[-1].weighted_sums)
        error_malignent = bce_prime(y, p[:, 0]) * gradient[0]
        error_benign = bce_prime(1 - y, p[:, 1]) * gradient[1]
        return np.column_stack((error_malignent, error_benign))

    def propagate_backwards(self, errors: np.ndarray) -> None:
        """calculates errors for all layers"""
        for index, error in enumerate(errors):
            weights = self.layers[-1].weights
            for layer in self.layers[::-1][1:]:
                error = layer.calculate_error(error, weights, index)
                weights = layer.weights

    def adjust_parameters(self, input_activation: np.ndarray) -> None:
        """applys learings for all the layers"""
        for index, input in enumerate(input_activation):
            prev_activation = input
            for layer in self.layers:
                layer.learn(self.rate, prev_activation, index)
                prev_activation = sigmoid(layer.weighted_sums)
        for layer in self.layers:
            layer.apply_changes(self.size)
