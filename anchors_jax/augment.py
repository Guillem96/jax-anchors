
from typing import Tuple

import jax
import jax.numpy as np

from anchors_jax.typing import Tensor


def random_lr_flip(rng: jax.random.PRNGKey,
                   image: Tensor,
                   boxes: Tensor,
                   prob: float = .5) -> Tuple[Tensor, Tensor]:
    """
    Parameters
    ----------
    rng: jax.random.PRNGKey
        JAX random key to ensure reproducibility
    image: Tensor of shape [H, W, C]
        Image tensor
    boxes: Tensor of shape [N, 4]
        Normalized boxes in range [0, 1]
    prob: float, default 0.5
        Probability to flip the image and boxes
    
    Returns
    -------
    Tuple[Tensor, Tensor]
        Flipped image and boxes
    """

    def lr_flip(o):
        image, boxes = o
        
        image = np.fliplr(image)

        # Flip the box
        x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)

        bb_w = x2 - x1
        delta_W = np.expand_dims(boxes[..., 0], axis=-1)

        x1 = 1. - delta_W - bb_w
        x2 = 1. - delta_W

        boxes = np.stack([x1, y1, x2, y2], axis=-1)
        boxes = boxes.reshape(-1, 4)

        return image, boxes

    return jax.lax.cond(jax.random.uniform(rng) < prob,
                        lr_flip, lambda o: o,
                        (image, boxes))


def rgb_to_grayscale(rng: jax.random.PRNGKey,
                     image: Tensor,
                     prob: float = .5) -> Tensor:

    def to_gray(image):
        dtype = image.dtype
        rgb_weights = np.array([0.2989, 0.5870, 0.1140]).reshape(1, 1, 3)
        image = np.sum(rgb_weights * image, axis=-1, keepdims=True)
        image = np.repeat(image, 3, axis=-1)
        return image.astype(dtype)

    return jax.lax.cond(jax.random.uniform(rng) < prob,
                        to_gray, lambda o: o, image)


def random_patch(rng: jax.random.PRNGKey,
                 image: Tensor,
                 boxes: Tensor,
                 crop_size: Tuple[int, int] = (300, 300),
                 min_patch_size: float = .5,
                 max_patch_size: float = 1.,
                 min_aspect_ratio: float = .5,
                 max_aspect_ratio: float = 2.) -> Tuple[Tensor, Tensor]:

    h, w, c = image.shape

    x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) / 2.
    cy = (y1 + y2) / 2.

    ar_key, patch_key, x_loc_key, y_loc_key = jax.random.split(rng, 4)

    ar = jax.random.uniform(ar_key, 
                            minval=min_aspect_ratio, 
                            maxval=max_aspect_ratio)

    patch_ratio = jax.random.uniform(patch_key,
                                     minval=min_patch_size, 
                                     maxval=max_patch_size)

    p_h = h * patch_ratio
    p_w = p_h * ar
    
    p_h = np.clip(p_h, a_max=h)
    p_w = np.clip(p_h, a_max=w)

    x = jax.random.uniform(x_loc_key, 
                           minval=0, 
                           maxval=w - p_w)

    y = jax.random.uniform(y_loc_key, 
                           minval=0, 
                           maxval=h - p_h)

    crop_box = np.array([x, y, x + p_w, y + p_h]).astype('int32')
    new_image = _crop_and_resize(image, crop_box, crop_size + (c,))

    keep_boxes = (cx > x) & (cx < (x + p_w))
    keep_boxes = keep_boxes & (cy > y) & (cy < (y + p_h))
    keep_boxes = np.repeat(keep_boxes, 4, axis=-1)
    keep_boxes = keep_boxes.astype('float32')

    x1 = x1 - x
    x2 = x1 + bw
    y1 = y1 - y
    y2 = y1 + bh

    x1 = np.clip(x1, a_min=0, a_max=p_w) / p_w
    x2 = np.clip(x2, a_min=0, a_max=p_w) / p_w
    y1 = np.clip(y1, a_min=0, a_max=p_h) / p_h
    y2 = np.clip(y2, a_min=0, a_max=p_h) / p_h

    boxes = np.concatenate([x1, y1, x2, y2], axis=-1)

    return new_image, boxes * keep_boxes


def _crop_and_resize(image: Tensor, 
                     box: Tensor, 
                     crop_size: Tuple[int, int, int]) -> Tensor:

    new_image = jax.lax.dynamic_slice(image, 
        np.array([box[1], box[0], 0]).reshape(3),
        np.array([box[3] - box[1], box[2] - box[0], 3]).reshape(3))

    return jax.image.resize(new_image, crop_size, 'bilinear')
