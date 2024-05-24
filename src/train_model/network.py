import numpy as np
import pandas as pd


class Network:

    def __init__(self, conf: dict, data: pd.DataFrame):
        self.layers = np.insert([30, 2], 1, conf["layer"])
        self.count = len(self.layers)
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])
        ]
        self.epochs = conf["epochs"]
        self.data = data

    def fit(self):
        for i in range(self.epochs):
            batch = self.data.loc[self.data["Batch"] == i // 8][2:].values
            prediction = self.forwardprop(batch)

    def forwardprop(self, batch: np.ndarray) -> np.ndarray:
        return np.array([])
