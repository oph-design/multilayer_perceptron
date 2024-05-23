import pandas as pd
import numpy as np


def get_indicies(batch_size: int, count: int) -> np.ndarray:
    """calculates the batch indexing"""
    res = []
    index = 0
    for i in range(1, count + 1, 1):
        res.append(index)
        if i % batch_size == 0:
            index += 1
    return np.array(res)


def index_batches(batch_size: int) -> pd.DataFrame:
    """loads training data and returns with batch numbers"""
    data = pd.DataFrame([])
    try:
        data = pd.read_csv("datasets/data_train.csv")
    except Exception as e:
        print(e)
        return data
    data = data.sample(frac=1).reset_index(drop=True)
    data.insert(0, "Batch", get_indicies(batch_size, data.shape[0]))
    return data
