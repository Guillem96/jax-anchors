from typing import Sequence

import jax.numpy as np
import haiku as hk

from haiku._src.initializers import _compute_fans

from anchors_jax.typing import Tensor


class XavierUniform(hk.initializers.Initializer):

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, shape: Sequence[int], dtype) -> Tensor:
        fan_in, fan_out = _compute_fans(shape)
        a = self.gain * np.sqrt(6. / (fan_in + fan_out))
        return hk.initializers.RandomUniform(-a, a)(shape, dtype)
