
from typing import List, Optional

import jax
from jax.experimental import stax

from anchors_jax import nn
from anchors_jax.typing import Layer

_CONF = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 
              'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def _make_layers(cfg, batch_norm: bool = True) -> List[Layer]:
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers.append(stax.MaxPool((2, 2), strides=(2, 2)))
        else:
            conv2d = stax.Conv(v, 
                               filter_shape=(3, 3), 
                               strides=(1, 1), 
                               padding='SAME')
            if batch_norm:
                layers += [conv2d, 
                           stax.BatchNorm(),
                           stax.Relu]
            else:
                layers += [conv2d, stax.Relu]
            in_channels = v

    return layers


def VGG16(num_classes: int = 1000, 
          include_top: bool = True,
          pooling: Optional[str] = None) -> Layer:
    assert pooling in {'avg', 'max', None}

    features = _make_layers(_CONF['VGG16'], False)

    if not include_top and pooling == 'avg':
        vgg = stax.serial(*features, nn.layers.GlobalAveragePooling())
    elif not include_top and pooling == 'max':
        vgg = stax.serial(*features, nn.layers.GlobalMaxPooling())
    elif not include_top and pooling is None:
        vgg = stax.serial(*features)
    else:
        classifier = [
            stax.Flatten,
            stax.Dense(4096), stax.Relu,
            stax.Dense(4096), stax.Relu,
            stax.Dense(num_classes), stax.Softmax
        ]
        vgg = stax.serial(*features, *classifier)

    return vgg


def VGG16_imagenet_weights(include_top: bool = True):
    from tensorflow.keras.applications import VGG16
    cfg = _CONF['VGG16']
    keras_vgg = VGG16(weights='imagenet', include_top=True)

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
