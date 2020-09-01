
from typing import Tuple

import jax.numpy as np


def GlobalAveragePooling(axis: Tuple[int, ...]):

    def init_fun(rng, input_shape):
        out_shape = list(input_shape)
        for a in axis:
            out_shape[a] = 1

        return tuple(out_shape), ()

    def apply_fun(params, x, **kwargs):
        return np.mean(x, axis=axis, keepdims=True)

    return init_fun, apply_fun


def GlobalMaxPooling(axis: Tuple[int, ...]):

    def init_fun(rng, input_shape):
        out_shape = list(input_shape)
        for a in axis:
            out_shape[a] = 1

        return tuple(out_shape), ()

    def apply_fun(params, x, **kwargs):
        return np.max(x, axis=axis, keepdims=True)

    return init_fun, apply_fun
