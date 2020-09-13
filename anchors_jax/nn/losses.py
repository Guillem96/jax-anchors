from typing import Iterable

import jax
import jax.numpy as np

from anchors_jax.typing import Tensor


def sparse_cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Examples
    --------
    >>> target = jax.random.randint(key, shape=(10,), maxval=3, minval=0)
    >>> predictions = jax.nn.softmax(jax.random.uniform(key, shape=(10, 3)))
    >>> loss = aj.nn.losses.sparse_cross_entropy(target, predictions)
    """
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true = jax.nn.one_hot(y_true, y_pred.shape[-1])
    return y_true * -np.log(y_pred)


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    epsilon = 1e-8
    y_pred = y_pred.reshape(-1)
    y_true = y_true.astype('float32')

    loss = y_true * np.log(y_pred + epsilon)
    loss += (1. - y_true) * np.log(1. - y_pred + epsilon)
    return -loss


def smooth_l1(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = y_true.reshape(-1, 4).astype('float32')
    y_pred = y_pred.reshape(-1, 4).astype('float32')

    error = y_true - y_pred
    abs_error = np.abs(error)
    mask = (abs_error < 1.).astype('float32')

    quad_error = np.power(error, 2) * .5
    lin_error = abs_error - .5

    return mask * quad_error + (1. - mask) * lin_error


def l2_loss(params: Iterable[Tensor]) -> Tensor:
  return 0.5 * sum(np.sum(np.square(p)) for p in params)
