import numpy as np
import pandas as pd
from .layer import Layer
from .functions import binary_cross_entropy


def format_array(source: np.ndarray) -> np.ndarray:
    """creates the opposite of the source array merges both"""
    opposite = 1 - source
    formated = np.empty((len(source) + len(opposite),), dtype=source.dtype)
    formated[0::2] = source
    formated[1::2] = opposite
    return formated


class Network:

    def __init__(self, conf: dict, data: pd.DataFrame):
        """initializes layers of the neural Network based on conf"""
        self.data = data
        self.epochs = conf["epochs"]
        self.rate = conf["learning_rate"]
        self.dim = np.insert([30, 2], 1, conf["layer"])
        self.count = len(self.dim)
        self.layers = [
            Layer(self.dim[x - 1], self.dim[x]) for x in range(1, self.count)
        ]

    def fit(self):
        """performs gradient descent for all layers in epoch time"""
        for i in range(self.epochs):
            batch = self.data.loc[self.data["Batch"] == i % 8].values
            y = format_array(batch[:, 1:2].flatten())
            x = batch[:, 2:]
            p = self.forwardprop(x).flatten()
            loss = np.mean(binary_cross_entropy(y, p))
            print(loss)

    def forwardprop(self, batch: np.ndarray) -> np.ndarray:
        """calculates the prediction for the current weights"""
        res = []
        for data in batch:
            for layer in self.layers:
                data = layer.get_neurons(data)
            res.append(data)
        return np.array(res)
