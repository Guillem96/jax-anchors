
from typing import List, Optional, Union

import jax
import haiku as hk

from anchors_jax import nn
from anchors_jax.typing import Tensor

_CONF = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 
              'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def _call_layers(
        cfg,
        inp: Tensor,
        batch_norm: bool = True, 
        include_top: bool = True,
        initial_weights: Optional[hk.Params] = None,
        output_feature_maps: bool = False) -> Union[Tensor, List[Tensor]]:

    x = inp
    partial_results = []

    # Ignore max pooling if we do not append the classifier
    if not include_top:
        cfg = cfg[:-1]
    
    i = 0
    base_name = 'vgg16/conv2_d'
    for v in cfg:
        if v == 'M':
            partial_results.append(x)
            x = hk.MaxPool(window_shape=2, strides=2, padding="VALID")(x)
        else:
            if i == 0:
                param_name = base_name
            else:
                param_name = base_name + f'_{i}'
            
            i += 1

            w_init = (None if initial_weights is None 
                      else hk.initializers.Constant(
                         constant=initial_weights[param_name]['w']))
            b_init = (None if initial_weights is None 
                      else hk.initializers.Constant(
                         constant=initial_weights[param_name]['b']))
            x = hk.Conv2D(v, kernel_shape=3, stride=1, padding='SAME', 
                          w_init=w_init, b_init=b_init)(x)

            if batch_norm:
                x = hk.BatchNorm(True, True, decay_rate=0.999)(x)

            x = jax.nn.relu(x)

    partial_results.append(x)

    if not output_feature_maps:
        return partial_results[-1]
    else:
        return partial_results


class VGG16(hk.Module):

    def __init__(self,
                 num_classes: int = 1000, 
                 include_top: bool = True,
                 pooling: Optional[str] = None,
                 pretrained: bool = True,
                 initial_weights=None,
                 output_feature_maps: bool = False) -> None:
        super(VGG16, self).__init__()

        assert pooling in {'avg', 'max', None}

        if include_top and output_feature_maps:
            raise ValueError("include_top to append a classifier on top of the "
                             "CNN in not compatible with outputing all the cnn "
                             "feature maps (enabled with output_feature_maps)")

        if output_feature_maps and pooling is not None:
            raise ValueError("Cannot set a pooling when outputing the"
                             " partial feature maps")

        self.num_classes = num_classes
        self.include_top = include_top
        self.pooling = pooling
        self.output_feature_maps = output_feature_maps

        if initial_weights is not None and pretrained:
            raise ValueError("When pretrained is True, initial_weights must"
                             " be None")

        if initial_weights is not None:
            self.initial_weights = initial_weights
        elif pretrained:
            self.initial_weights = VGG16_imagenet_weights(include_top)
        else:
            self.initial_weights = None

    def features(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        return _call_layers(_CONF['VGG16'],
                            inp=x,
                            batch_norm=False,
                            include_top=self.include_top,
                            initial_weights=self.initial_weights,
                            output_feature_maps=self.output_feature_maps)

    def _classifier(self, x: Tensor) -> Tensor:
        override_weights = (self.initial_weights is not None and 
                            self.num_classes == 1000)

        base_name = 'vgg16/linear'
        cfg = [4096, 4096, self.num_classes]
        for i, f in enumerate(cfg):
            if i == 0:
                param_name = base_name
            else:
                param_name = base_name + f'_{i}'

            w_init = (None if not override_weights
                      else hk.initializers.Constant(
                          constant=self.initial_weights[param_name]['w']))
            b_init = (None if not override_weights
                      else hk.initializers.Constant(
                          constant=self.initial_weights[param_name]['b']))
            x = hk.Linear(f, w_init=w_init, b_init=b_init)(x)
            if i < len(cfg) - 1:
                x = jax.nn.relu(x)
            else:
                x = jax.nn.softmax(x)
        return x

    def __call__(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        x = self.features(x)

        if not self.include_top and self.pooling == 'avg':
            x = nn.layers.GlobalAveragePooling()(x)
        elif not self.include_top and self.pooling == 'max':
            x = nn.layers.GlobalMaxPooling()(x)
        elif not self.include_top and self.pooling is None:
            x = x
        else:
            x = hk.Flatten()(x)
            x = self._classifier(x)

        return x


def VGG16_imagenet_weights(include_top: bool = True):
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16

    with tf.device('/cpu:0'):
        keras_vgg = VGG16(weights='imagenet', include_top=include_top)

    weights = {}
    clf_i = 0
    features_i = 0

    features_base_name = 'vgg16/conv2_d'
    clf_base_name = 'vgg16/linear'

    for i, it in enumerate(range(0, len(keras_vgg.variables), 2)):
        is_feature = 'conv' in keras_vgg.variables[it].name
        is_clf = ('fc' in keras_vgg.variables[it].name or 
                  'predictions' in keras_vgg.variables[it].name)

        if is_feature and features_i == 0:
            name = features_base_name
            features_i += 1
        elif is_feature:
            name = features_base_name + f'_{features_i}'
            features_i += 1
        elif is_clf and clf_i == 0:
            clf_i += 1
            name = clf_base_name
        elif is_clf:
            name = clf_base_name + f'_{clf_i}'
            clf_i += 1
        else:
            raise ValueError(f'Unexpected parameter from Keras VGG16: '
                             f'{keras_vgg.variables[it].name}')

        w = jax.device_put(keras_vgg.variables[it].numpy())
        b = jax.device_put(keras_vgg.variables[it + 1].numpy())
        weights[name] = {'w': w, 'b': b}

    return weights
