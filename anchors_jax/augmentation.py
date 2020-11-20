
from typing import Tensor

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
    if jax.random.uniform(rng, shape=[]) < prob:
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

    if jax.random.uniform(rng, shape=[]) < prob:
        rgb_weights = np.array([0.2989, 0.5870, 0.1140]).reshape(1, 1, 3)
        image = np.sum(rgb_weights * image, axis=-1, keepdims=True)
        image = np.repeat(image, 3, axis=-1)

    return image