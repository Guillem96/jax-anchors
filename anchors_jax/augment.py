
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
    if jax.random.uniform(rng) < prob:
        image = np.fliplr(image)

        # Flip the box
        x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)

        bb_w = x2 - x1
        delta_W = np.expand_dims(boxes[:, 0], axis=-1)

        x1 = 1. - delta_W - bb_w
        x2 = 1. - delta_W

        boxes = np.stack([x1, y1, x2, y2], axis=-1)
        boxes = boxes.reshape(-1, 4)

    return image, boxes


def rgb_to_grayscale(rng: jax.random.PRNGKey,
                     image: Tensor,
                     prob: float = .5) -> Tensor:

    if jax.random.uniform(rng) < prob:
        rgb_weights = np.array([0.2989, 0.5870, 0.1140]).reshape(1, 1, 3)
        image = np.sum(rgb_weights * image, axis=-1, keepdims=True)
        image = np.repeat(image, 3, axis=-1)

    return image


def random_patch(rng: jax.random.PRNGKey,
                 image: Tensor,
                 boxes: Tensor,
                 min_patch_size: float = .5,
                 max_patch_size: float = 1.,
                 min_aspect_ratio: float = .5,
                 max_aspect_ratio: float = 2.) -> Tuple[Tensor, Tensor]:

    h, w, _ = image.shape

    x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h

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

    x = jax.random.uniform(x_loc_key, 
                           minval=0, 
                           maxval=w - p_w)

    y = jax.random.uniform(y_loc_key, 
                           minval=0, 
                           maxval=h - p_h)

    new_image = image[y.astype('int32'): (y + p_h).astype('int32'), 
                      x.astype('int32'): (x + p_w).astype('int32')]

    keep_boxes = (cx > x) & (cx < (x + p_w))
    keep_boxes = keep_boxes & (cy > y) & (cy < (y + p_h))
    # keep_boxes = np.expand_dims(keep_boxes, -1)
    keep_boxes = np.repeat(keep_boxes, 4, axis=-1)
    keep_boxes = keep_boxes.astype('float32')

    x1 = np.clip(x1 - x, a_min=0, a_max=p_w) / p_w
    x2 = np.clip(x2 - x, a_min=0, a_max=p_w) / p_w
    y1 = np.clip(y1 - y, a_min=0, a_max=p_h) / p_h
    y2 = np.clip(y2 - y, a_min=0, a_max=p_h) / p_h

    boxes = np.concatenate([x1, y1, x2, y2], axis=-1)

    return new_image, boxes * keep_boxes

