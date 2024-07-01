import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """applys sigmoid activation on array"""
    return 1 / (1 + np.exp(x * -1))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    """applys sigmoid derivative on array"""
    return sigmoid(x) * (1 - sigmoid(x))


def soft_max(x: np.ndarray) -> np.ndarray:
    """applys soft_max activation on array"""
    return np.exp(x) / np.sum(np.exp(x))


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """calculates error for the current prediction"""
    return -(y * np.log(p) + (1 - y) * np.log(p))


def bce_prime(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """calculates error derivative for the current prediction"""
    return (p - y) / p * (1 - p)
