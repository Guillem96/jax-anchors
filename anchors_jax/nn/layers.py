
from typing import Tuple

import jax
import jax.numpy as np

import haiku as hk

from anchors_jax.typing import Tensor


class GlobalAveragePooling(hk.Module):

    def __init__(axis: Tuple[int, ...]):
        super(GlobalAveragePooling, self).__init__()
        self.axis = axis 

    def __call__(self, x):
        return np.mean(x, axis=axis, keepdims=True)


class GlobalMaxPooling(hk.Module):

    def __init__(axis: Tuple[int, ...]):
        super(GlobalMaxPooling, self).__init__()
        self.axis = axis 

    def __call__(self, x):
        return np.max(x, axis=axis, keepdims=True)


class FasterRCNNRPN(hk.Module):

    def __init__(self, k: int, features: int):
        super(FasterRCNNRPN, self).__init__()

        self.k = k
        self.features = features
    
    def __call__(self, x: Tensor) -> Tensor:
        bias_init = hk.initializers.Constant(0.)
        kernel_init = hk.initializers.RandomNormal(stddev=0.01)

        x = hk.Conv2D(output_channels=self.features,
                      kernel_shape=3, stride=1,
                      w_init=kernel_init,
                      b_init=bias_init,
                      padding="SAME")(x)
        x = jax.nn.relu(x)

        cls_logits = hk.Conv2D(output_channels=self.k, 
                               kernel_shape=1, stride=1,
                               w_init=kernel_init,
                               b_init=bias_init,
                               padding="SAME")(x)
        cls_logits = jax.nn.sigmoid(cls_logits)
        cls_logits = cls_logits.reshape(x.shape[0], -1)

        reg_logits = hk.Conv2D(output_channels=self.k * 4, 
                               kernel_shape=1, stride=1,
                               w_init=kernel_init,
                               b_init=bias_init,
                               padding="SAME")(x)
        reg_logits = reg_logits.reshape(x.shape[0], -1, 4)

        return cls_logits, reg_logits
