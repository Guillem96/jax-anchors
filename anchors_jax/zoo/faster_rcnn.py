import jax
import haiku as hk

import anchors_jax as aj
from anchors_jax.typing import Tensor


class FasterRCNN(hk.Module):
    
    def __init__(self,
                 num_classes: int, 
                 k: int,
                 pretrained_backbone: bool = True):
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes
        self.k = k
        self.pretrained_backbone = pretrained_backbone

    def __call__(self, x: Tensor) -> Tensor:
        x = aj.zoo.VGG16(include_top=False, 
                         pretrained=self.pretrained_backbone)(x)
        return aj.nn.layers.FasterRCNNRPN(features=256, k=self.k)(x)
