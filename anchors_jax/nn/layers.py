
from typing import Tuple

import jax
import jax.numpy as np
from jax.experimental import stax

from anchors_jax.typing import Layer


def GlobalAveragePooling(axis: Tuple[int, ...]) -> Layer:

    def init_fun(rng, input_shape):
        out_shape = list(input_shape)
        for a in axis:
            out_shape[a] = 1

        return tuple(out_shape), ()

    def apply_fun(params, x, **kwargs):
        return np.mean(x, axis=axis, keepdims=True)

    return init_fun, apply_fun


def GlobalMaxPooling(axis: Tuple[int, ...]) -> Layer:

    def init_fun(rng, input_shape):
        out_shape = list(input_shape)
        for a in axis:
            out_shape[a] = 1

        return tuple(out_shape), ()

    def apply_fun(params, x, **kwargs):
        return np.max(x, axis=axis, keepdims=True)

    return init_fun, apply_fun


def FasterRCNNRPN(k: int, num_classes: int, features: int) -> Layer:

    conv_init_fn, conv_forward_fn = stax.Conv(features, 
                                              filter_shape=(3, 3), 
                                              strides=(1, 1),
                                              padding='SAME')

    head_init_fn, head_forward_fn = stax.parallel(
        stax.Conv(k * num_classes, 
                  filter_shape=(1, 1),
                  strides=(1, 1),
                  padding='SAME'),
        stax.Conv(k * 4, 
                  filter_shape=(1, 1),
                  strides=(1, 1),
                  padding='SAME'))

    def init_fun(rng, input_shape):
        conv_rng, head_rng = jax.random.split(rng, 2)
        input_shape, conv_params = conv_init_fn(conv_rng, input_shape)
        output_shape, head_params = head_init_fn(head_rng, [input_shape] * 2)
        return output_shape, (conv_params, head_params)

    def apply_fun(params, x, **kwargs):
        conv_params, head_params = params
        x = conv_forward_fn(conv_params, x, **kwargs)
        x = jax.nn.relu(x)

        cls_logits, reg_logits = head_forward_fn(head_params, [x, x], **kwargs)
        cls_logits = cls_logits.reshape(x.shape[0], -1, num_classes)
        cls_logits = jax.nn.softmax(cls_logits, axis=-1)

        reg_logits = reg_logits.reshape(x.shape[0], -1, 4)

        return cls_logits, reg_logits

    return init_fun, apply_fun
