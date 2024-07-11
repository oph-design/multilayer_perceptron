import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .visualizer import Visualizer
from .mathematics import binary_cross_entropy, accuracy, calc_error_gradient, r_squared


class Network:
    def __init__(self, data_train, data_val, layers: list, conf: dict | None):
        """initializes layers of the neural Network based on conf"""
        self.layers = layers
        self.data_t = data_train
        self.data_v = data_val
        if conf is not None:
            self.epochs = conf["epochs"]
            self.rate = conf["learning_rate"]
            self.size = conf["batch_size"]
            self.index = [0, 0]
            figure, axes = plt.subplots(1, 2)
            self.accuracy = Visualizer(self.epochs, axes[0], "Accuracy")
            self.loss = Visualizer(self.epochs, axes[1], "Loss")
            plt.ion()

    def __del__(self):
        plt.close("all")

    def get_batch(self, data: pd.DataFrame, id: int) -> np.ndarray:
        index = self.index[id]
        limit = int(data.shape[0] / self.size) if data.shape[0] >= self.size else 1
        self.index[id] = self.index[id] + 1 if self.index[id] + 1 != limit else 0
        return data.loc[data["Batch"] == index].values

    def fit(self):
        """performs gradient descent for all layers in epoch time"""
        print(f"x_train shape: {self.data_t.shape[0]}, {self.data_t.shape[1] - 2}")
        print(f"x_valid shape: {self.data_v.shape[0]}, {self.data_v.shape[1] - 2}")
        for i in range(self.epochs):
            batch = self.get_batch(self.data_t, 0)
            validate = self.get_batch(self.data_v, 1)
            y_val = validate[:, 1:2].flatten()
            p_val = self.make_prediction(validate[:, 2:])
            target = batch[:, 1:2].flatten()
            prediction = self.make_prediction(batch[:, 2:])
            print(np.column_stack((prediction, target, 1 - target)))
            self.status(i, prediction, target, p_val, y_val)
            self.propagate_backwards(calc_error_gradient(target, prediction))
            self.adjust_parameters(batch[:, 2:])

    def make_prediction(self, batch: np.ndarray) -> np.ndarray:
        """calculates the prediction for the current weights"""
        res = []
        for index, data in enumerate(batch):
            for layer in self.layers:
                data = layer.get_neurons(data, index)
            res.append(data)
        return np.array(res)

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
                prev_activation = layer.activations[index]
        for layer in self.layers:
            layer.apply_changes(self.size)

    def status(self, i: int, p_train, y_train, p_val, y_val) -> None:
        """prints epochs status and advances the accuracy and loss graphs"""
        e_train = np.mean(binary_cross_entropy(y_train, p_train.T[0]))
        e_val = np.mean(binary_cross_entropy(y_val, p_val.T[0]))
        count = str(i + 1) if i + 1 > 9 else "0" + str(i + 1)
        self.loss.plot_data(e_train, e_val)
        self.accuracy.plot_data(accuracy(y_train, p_train), accuracy(y_val, p_val))
        print(
            f"epoch {count}/{self.epochs} - loss: {e_train:.4e} - val_loss: {e_val:.4e}"
        )
        self.accuracy.draw_plot()
        self.loss.draw_plot()

    def save_to_file(self):
        name = input("name your model: ")
        file = "results/models/" + name + ".npz"
        network_dict = {}
        for i, layer in enumerate(self.layers):
            network_dict[f"weights_l{i}"] = layer.weights
            network_dict[f"biases_l{i}"] = layer.biases
        np.savez(file, **network_dict)
        print(f"> saving model '{name}' to ./results/models ...")

    def evalulate_model(self, name: str):
        y = self.data_v.values[:, 0:1]
        p = self.make_prediction(self.data_v.values[:, 1:])
        rounded = np.where(p >= 0.5, np.ceil(p), np.floor(p))
        result = np.column_stack((y, rounded.T[0])).astype(str)
        result[result == str(1.0)] = "M"
        result[result == str(0.0)] = "B"
        error = np.mean(binary_cross_entropy(y, p.T[0]))
        accur = accuracy(y, p) * 100
        r_sq = r_squared(y, p.T[0])
        print(f"Prediction Error:       {error}")
        print(f"Prediction R Squared:   {r_sq}")
        print(f"Prediction Accuracy:    {accur}%")
        print("> saving prediction to ./results/evals ...")
        file = pd.DataFrame(result, columns=["Truth", "Prediction"])
        file.to_csv("results/evals/" + name + ".csv")
