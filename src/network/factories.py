import numpy as np
from .layer import Layer


def train_factory(dim: np.ndarray, batch: int) -> list:
    return [
        Layer(
            np.random.uniform(-0.1, 0.1, size=(dim[x], dim[x - 1])),
            np.random.uniform(-0.1, 0.1, size=(dim[x])),
            batch,
            x % len(dim),
        )
        for x in range(1, len(dim))
    ]


def predict_factory(map: dict, length: int) -> list:
    dim = int(len(map.keys()) / 2)
    return [
        Layer(
            map[f"weights_l{x}"],
            map[f"biases_l{x}"],
            length,
            x % dim,
        )
        for x in range(0, dim)
    ]
