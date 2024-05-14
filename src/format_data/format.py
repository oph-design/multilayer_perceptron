import pandas as pd
from pandas.core.api import Series
from .columns import colnames


def normalize(column: Series) -> Series:
    """calculates normalized values for each column"""
    return (column - column.min()) / (column.max() - column.min())


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """normalizes_data to values between 0 and 1"""
    classfier = data["diagnosis"]
    features = data.iloc[:, 2:]
    normalized_features = features.apply(normalize)
    return pd.concat([classfier, normalized_features], axis=1)


def label_data(data: pd.DataFrame) -> pd.DataFrame:
    """swapes house values out for 1s and 0s"""
    res = data.copy()
    res["diagnosis"] = (res["diagnosis"] == "M").astype(int)
    return res


def split_data(location: str, split: float) -> None:
    """splits data in test and train data"""
    data = pd.read_csv(location, names=colnames, header=None)
    data = normalize_data(label_data(data))
    test_data = data.sample(n=(int(data.shape[0] * (1 - split))))
    train_data = data.drop(index=list(test_data.index))
    test_data.to_csv("datasets/data_test.csv", index=False)
    train_data.to_csv("datasets/data_train.csv", index=False)


if __name__ == "__main__":
    split_data("ressources/data.csv", 0.8)
