
from typing import List, Optional

import jax
import haiku as hk

from anchors_jax import nn
from anchors_jax.typing import Tensor

_CONF = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 
              'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def _call_layers(cfg,
                 inp: Tensor,
                 batch_norm: bool = True, 
                 include_top: bool = True) -> List[hk.Module]:
    x = inp
    in_channels = 3
    
    # Ignore max pooling if we do not append the classifier
    if not include_top:
        cfg = cfg[:-1]

    for v in cfg:
        if v == 'M':
            x = hk.MaxPool(window_shape=[2, 2], 
                           strides=[2, 2],
                           padding="VALID")(x)
        else:
            x = hk.Conv2D(v, 
                          kernel_shape=[3, 3], 
                          stride=1, 
                          padding='SAME')(x)

            if batch_norm:
                x = hk.BatchNorm(True, True, decay_rate=0.999)(x)

            x = jax.nn.relu(x)
            in_channels = v

    return x


class VGG16(hk.Module):
    
    def __init__(self,
                 num_classes: int = 1000, 
                 include_top: bool = True,
                 pooling: Optional[str] = None) -> None:
        super(VGG16, self).__init__()

        assert pooling in {'avg', 'max', None}

        self.num_classes = num_classes
        self.include_top = include_top
        self.pooling = pooling

    def __call__(self, x: Tensor) -> Tensor:
        x = _call_layers(_CONF['VGG16'], x, False, self.include_top)

        if not self.include_top and self.pooling == 'avg':
            x = nn.layers.GlobalAveragePooling()(x)
        elif not self.include_top and self.pooling == 'max':
            x = nn.layers.GlobalMaxPooling()(x)
        else:
            x = hk.Flatten()(x)
            
            x = hk.Linear(4096)(x)
            x = jax.nn.relu(x)

            x = hk.Linear(4096)(x)
            x = jax.nn.relu(x)
            
            x = hk.Linear(num_classes)
            x = jax.nn.softmax(x)

        return x


def VGG16_imagenet_weights(include_top: bool = True):
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    cfg = _CONF['VGG16']
    with tf.device('/cpu:0'):
        keras_vgg = VGG16(weights='imagenet', include_top=include_top)

    # Ignore max pooling if we do not append the classifier
    if not include_top:
        cfg = cfg[:-1]

    params = []
    i = 0
    for c in cfg:
        if c == 'M':
            params.append(tuple())
        else:
            params.append(tuple([jax.device_put(keras_vgg.variables[it].numpy())
                                 for it in range(i, i + 2)])) # Conv kernel + bias
            params.append(tuple()) # Conv activation
            i += 2

    if include_top:
        params.append(tuple()) # Flatten
        for it in range(i, len(keras_vgg.variables), 2):
            params.append(tuple([jax.device_put(keras_vgg.variables[it].numpy())
                                for it in range(it, it + 2)]))
            params.append(tuple())

    return params
