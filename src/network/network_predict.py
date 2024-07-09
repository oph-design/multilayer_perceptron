import pandas as pd
import numpy as np
from .network_base import Network_Base
from .layer import Layer


class Network_Predict(Network_Base):
    def __init__(self, map: dict, data_train: pd.DataFrame, data_test: pd.DataFrame):
        size = len(data_train) + len(data_test)
        super().__init__(data_train, data_test, size, np.array([30, 2]))
        self.layers = self.transform_layers(map, int(len(map.keys()) / 2))

    def transform_layers(self, map: dict, length: int) -> list:
        return [
            Layer(
                map[f"weights_{x}"],
                map[f"biases_{x}"],
                self.size,
                x % length,
            )
            for x in range(0, length)
        ]
