import jax
import jax.numpy as np

from anchors_jax.typing import Image, Tensor


def caffe_preprocess(im: Image) -> Tensor:
    mean = [103.939, 116.779, 123.68]

    im = np.array(im).astype('float32')
    im = np.flip(im, axis=-1)
    im = jax.ops.index_add(im, jax.ops.index[..., 0], -mean[0])
    im = jax.ops.index_add(im, jax.ops.index[..., 1], -mean[1])
    im = jax.ops.index_add(im, jax.ops.index[..., 2], -mean[2])

    return im
