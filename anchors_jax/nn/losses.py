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
    return -np.sum(y_true * np.log(y_pred), -1)


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    epsilon = 1e-8

    y_true = y_true.astype('float32')
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = y_true * np.log(y_pred + epsilon)
    loss += (1. - y_true) * np.log(1. - y_pred + epsilon)
    return -loss


def focal_loss(y_true: Tensor, y_pred: Tensor, 
               alpha: float = 0.25, gamma: float = 2.0) -> Tensor:

    alpha_factor = np.ones_like(y_true) * alpha
    alpha_factor = np.where(y_true > 0, alpha_factor, 1 - alpha_factor)
    focal_weight = np.where(y_true > 0, 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma

    return focal_weight * binary_cross_entropy(y_true, y_pred)


def smooth_l1(y_true: Tensor, y_pred: Tensor, beta: float = 1.) -> Tensor:
    y_true = y_true.astype('float32')
    y_pred = y_pred.astype('float32')

    error = y_true - y_pred
    abs_error = np.abs(error)
    return np.where(abs_error < beta,
                    .5 * abs_error ** 2 / beta,
                    abs_error - .5 * beta)


def l2_loss(params: Iterable[Tensor]) -> Tensor:
  return 0.5 * sum(np.sum(np.square(p)) for p in params)
