import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """applys sigmoid activation on array"""
    return 1 / (1 + np.exp(x * -1))


def soft_max(x: np.ndarray) -> np.ndarray:
    """applys soft_max activation on array"""
    return np.exp(x) / np.sum(np.exp(x))


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    """calculates error for the current prediction"""
    return float(1 / len(y) * np.sum(y * np.log(p) + (1 - y) * np.log(p)))
