from typing import Tuple
import numpy as np

def standart_scale(x: np.ndarray, mean_=None, min_=None) -> Tuple[np.ndarray, float, float]:
    mean_ = np.mean(x, axis=0) if mean_ is None else mean_
    min_ = np.min(x, axis=0) if min_ is None else min_

    return (x - min_) / mean_, mean_, min_

def standart_scale_inverse(x: np.ndarray, mean_, min_) -> np.ndarray:
    return x * mean_ + min_