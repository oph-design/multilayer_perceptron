import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """applys sigmoid activation on array"""
    return 1 / (1 + np.exp(x * -1))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    """applys sigmoid derivative on array"""
    return sigmoid(x) * (1 - sigmoid(x))


def soft_max(x: np.ndarray) -> np.ndarray:
    """applys soft_max activation on array"""
    return np.exp(x - np.max(x)) / np.sum(np.exp(x))


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """calculates error for the current prediction"""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calc_error_gradient(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """calculates the error in the output layer"""
    return np.column_stack((y - p[:, 0], (1 - y) - p[:, 1]))


def accuracy(y: np.ndarray, p: np.ndarray) -> np.float_:
    rounded = np.where(p >= 0.5, np.ceil(p), np.floor(p))
    return np.mean(np.argmax(rounded, axis=1) == y)


def r_squared(y: np.ndarray, p: np.ndarray) -> np.float_:
    return 1 - (np.sum(np.power(y - p, 2)) / np.sum(np.power(y - np.mean(y), 2)))
