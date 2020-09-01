import jax.numpy as np

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
    epsilon = 1e-6
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true = jax.nn.one_hot(y_true, y_pred.shape[-1])
    return y_true * -np.log(y_pred)


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    epsilon = 1e-6
    y_pred = y_pred.reshape(-1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    y_true = y_true.astype('float32')

    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    return -np.log(pt)


def smooth_l1(y_true: Tensor, y_pred: Tensor) -> Tensor:
    error = y_true - y_pred
    abs_error = np.abs(error)

    return np.where(abs_error < 1., 
                    np.power(error, 2) * .5,
                    abs_error - .5)
