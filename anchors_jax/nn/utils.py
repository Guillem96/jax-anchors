import jax

from anchors_jax.typing import Tensor


def one_hot(labels: Tensor, depth: int) -> Tensor:
    """
    Object detection dedicated one hot encoding that encodes the 0s as
    a vector full of zeros and the remaining labels as a usual one hot vector

    Examples
    --------
    >>> labels = np.array([0, 1, 2])
    >>> aj.nn.utils.one_hot(labels, 3)
    [[0., 0.], 
     [1., 0.], 
     [0., 1.]] 
    """
    ohe = jax.nn.one_hot(labels, depth)
    return ohe[..., 1:]
