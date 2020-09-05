import jax
import haiku as hk

import anchors_jax as aj
from anchors_jax.typing import Tensor


class FasterRCNN(hk.Module):
    
    def __init__(self,
                 num_classes: int, 
                 k: int):
        self.num_classes = num_classes
        self.k = k

    def __call__(self, x: Tensor) -> Tensor:
        x = aj.zoo.VGG16(include_top=False)(x)
        x = jax.nn.relu(x)
        return aj.nn.layers.FasterRCNNRPN(features=256, 
                                          num_classes=self.num_classes, 
                                          k=self.k)(x)
