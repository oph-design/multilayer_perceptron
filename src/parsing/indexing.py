import pandas as pd
import numpy as np


def read_data(path: str) -> pd.DataFrame:
    data = pd.DataFrame([])
    try:
        data = pd.read_csv(path)
    except Exception as e:
        print(e)
    return data


def get_indicies(batch_size: int, count: int) -> np.ndarray:
    """calculates the batch indexing"""
    res = []
    index = 0
    for i in range(1, count + 1, 1):
        res.append(index)
        if i % batch_size == 0:
            index += 1
    return np.array(res)


def indexing(path: str, batch_size: int) -> pd.DataFrame:
    """loads training data and returns with batch numbers"""
    data = read_data(path)
    data = data.sample(frac=1).reset_index(drop=True)
    data.insert(0, "Batch", get_indicies(batch_size, data.shape[0]))
    return data
