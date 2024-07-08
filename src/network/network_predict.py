import pandas as pd
import numpy as np
from .network_base import Network_Base


class Network_Predict(Network_Base):
    def __init__(self, map: dict, data_train: pd.DataFrame, data_test: pd.DataFrame):
        size = len(data_train) + len(data_test)
        super().__init__(data_train, data_test, size, np.array([30, 2]))
