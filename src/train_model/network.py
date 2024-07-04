import numpy as np
import pandas as pd
from .layer import Layer
from .mathematics import bce_prime, cumulative_error, sigmoid_prime, sigmoid


class Network:

    def __init__(self, conf: dict, data_train: pd.DataFrame, data_test: pd.DataFrame):
        """initializes layers of the neural Network based on conf"""
        self.data_t = data_train
        self.data_v = data_test
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
        print(f"x_train shape: {self.data_t.shape[0]}, {self.data_t.shape[1] - 2}")
        print(f"x_valid shape: {self.data_v.shape[0]}, {self.data_t.shape[1] - 2}")
        for i in range(self.epochs):
            batch = self.data_t.loc[self.data_t["Batch"] == i % self.size].values
            target = batch[:, 1:2].flatten()
            input = batch[:, 2:]
            prediction = self.feed_forward(input)
            self.print_status(i, prediction, target)
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

    def print_status(self, i: int, p_train: np.ndarray, y_train: np.ndarray) -> None:
        validate = self.data_v.loc[self.data_v["Batch"] == i % self.size].values
        p_val = self.feed_forward(validate[:, 2:])
        e_train = str(cumulative_error(y_train, p_train.T))[:6]
        e_val = str(cumulative_error(validate[:, 1:2].flatten(), p_val.T))[:6]
        count = str(i + 1) if i + 1 > 9 else "0" + str(i + 1)
        print(f"epoch {count}/{self.epochs} - loss: {e_train} - val_loss: {e_val}")
