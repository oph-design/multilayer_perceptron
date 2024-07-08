import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .layer import Layer
from .visualizer import Visualizer
from .mathematics import binary_cross_entropy, accuracy


class Network:
    def __init__(self, conf: dict, data_train: pd.DataFrame, data_test: pd.DataFrame):
        """initializes layers of the neural Network based on conf"""
        plt.ion()
        figure, axis = plt.subplots(1, 2)
        dim = np.insert([30, 2], 1, conf["layer"])
        self.data_t = data_train
        self.data_v = data_test
        self.epochs = conf["epochs"]
        self.rate = conf["learning_rate"]
        self.size = conf["batch_size"]
        self.accuracy = Visualizer(self.epochs, axis[0], "Accuracy")
        self.loss = Visualizer(self.epochs, axis[1], "Loss")
        self.layers = [
            Layer(dim[x - 1], dim[x], self.size, x % len(dim))
            for x in range(1, len(dim))
        ]

    def __del__(self):
        plt.close("all")

    def fit(self):
        """performs gradient descent for all layers in epoch time"""
        print(f"x_train shape: {self.data_t.shape[0]}, {self.data_t.shape[1] - 2}")
        print(f"x_valid shape: {self.data_v.shape[0]}, {self.data_t.shape[1] - 2}")
        for i in range(self.epochs):
            batch = self.data_t.loc[self.data_t["Batch"] == i % self.size].values
            validate = self.data_v.loc[self.data_v["Batch"] == i % self.size].values
            y_val = validate[:, 1:2].flatten()
            p_val = self.feed_forward(validate[:, 2:])
            target = batch[:, 1:2].flatten()
            input = batch[:, 2:]
            prediction = self.feed_forward(input)
            self.status(i, prediction, target, p_val, y_val)
            error = self.calc_out_layer_error(target, prediction)
            self.propagate_backwards(error)
            self.adjust_parameters(input)

    def feed_forward(self, batch: np.ndarray) -> np.ndarray:
        """calculates the prediction for the current weights"""
        res = []
        for index, data in enumerate(batch):
            for layer in self.layers:
                data = layer.get_neurons(data, index)
            res.append(data)
        return np.array(res)

    def calc_out_layer_error(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """calculates the error in the output layer"""
        error_malignent = y - p[:, 0]
        print(y)
        print(1 - y)
        print(p[:, 0])
        print(p[:, 1])
        error_benign = (1 - y) - p[:, 1]
        return np.column_stack((error_benign, error_malignent))

    def propagate_backwards(self, errors: np.ndarray) -> None:
        """calculates errors for all layers"""
        self.layers[-1].errors = errors
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
                prev_activation = layer.get_activation(index)
        for layer in self.layers:
            layer.apply_changes(self.size)

    def status(self, i: int, p_train, y_train, p_val, y_val) -> None:
        """prints epochs status and advances the accuracy and loss graphs"""
        e_train = np.mean(binary_cross_entropy(y_train, p_train.T[0]))
        e_val = np.mean(binary_cross_entropy(y_val, p_val.T[0]))
        count = str(i + 1) if i + 1 > 9 else "0" + str(i + 1)
        self.loss.plot_data(e_train, e_val)
        self.accuracy.plot_data(accuracy(y_train, p_train), accuracy(y_val, p_val))
        print(f"epoch {count}/{self.epochs} - loss: {e_train} - val_loss: {e_val}")
        self.accuracy.draw_plot()

    def save_to_file(self):
        name = input("name your model: ")
        file = "models/" + name + ".npz"
        network_dict = {}
        for i, layer in enumerate(self.layers):
            network_dict[f"weights_l{i}"] = layer.weights
            network_dict[f"biases_l{i}"] = layer.biases
        np.savez(file, **network_dict)
        print(f"> saving model '{name}' to ./models ...")
