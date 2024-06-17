import numpy as np
import pandas as pd


class Layer:

    def __init__(self, prev: int, count: int):
        self.biases = np.array([np.random.randn() for _ in range(count)])
        self.weights = np.array(
            [[np.random.rand() for _ in range(prev)] for _ in range(count)]
        )

    def get_neurons(self, prev_neurons: np.ndarray) -> np.ndarray:
        activation = np.dot(self.weights, prev_neurons)
        activation = activation + self.biases
        return 1 / (1 + np.exp(activation * -1))


class Network:

    def __init__(self, conf: dict, data: pd.DataFrame):
        self.shape = np.insert([30, 2], 1, conf["layer"])
        self.count = len(self.shape)
        self.layers = [
            Layer(self.shape[x - 1], self.shape[x]) for x in range(1, self.count)
        ]
        self.epochs = conf["epochs"]
        self.data = data

    def fit(self):
        for i in range(self.epochs):
            batch = self.data.loc[self.data["Batch"] == i // 8].values
            batch = batch[:, 2:]
            prediction = self.forwardprop(batch)
            print(prediction)

    def forwardprop(self, batch: np.ndarray) -> np.ndarray:
        prediction = np.array([0, 0])
        for data in batch:
            for layer in self.layers:
                data = layer.get_neurons(data)
            prediction = prediction + data
        return prediction / batch.shape[1]
