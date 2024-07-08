import numpy as np
import pandas as pd
from .mathematics import xavier_init
from .layer import Layer


class Network_Base:
    def __init__(
        self,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        size: int,
        dim: np.ndarray,
    ):
        self.data_t = data_train
        self.data_v = data_test
        self.size = size
        self.layers = self.init_layers(dim)

    def feed_forward(self, batch: np.ndarray) -> np.ndarray:
        """calculates the prediction for the current weights"""
        res = []
        for index, data in enumerate(batch):
            for layer in self.layers:
                data = layer.get_neurons(data, index)
            res.append(data)
        return np.array(res)

    def init_layers(self, dim: np.ndarray) -> list:
        return [
            Layer(
                xavier_init(dim[x], dim[x - 1]),
                xavier_init(dim[x]).flatten(),
                self.size,
                x % len(dim),
            )
            for x in range(1, len(dim))
        ]
