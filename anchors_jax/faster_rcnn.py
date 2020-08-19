from typing import Tuple, Sequence

import jax
import jax.numpy as np

from anchors_jax.typing import Tensor


def single_point_anchors(
        aspect_ratios: Sequence[float] = (.5, 1., 2.),
        scales: Sequence[float] = (512**2, 256**2, 128**2)) -> Tensor:
    
    """
    Generates a single set of K anchors which is center is located at the origin

    Parameters
    ----------
    aspect_rations: Sequence[float], default (.5, 1., 2.)
        Different aspect ratios for each anchor
    scales: Sequence[float], default (512**2, 256**2, 128**2)
        Different scales for each anchor.
    
    Returns
    -------
    Tensor of shape [K, 4]
        Where K is equal to len(aspect_ratios) * len(scales). Note that the 
        boxes format is (x_min, y_min, m_max, y_max). Since the center is located
        at the origin the resulting coordinates of each box will correspond to
          - x_min = -box_width / 2
          - y_min = -box_height / 2
          - x_max = box_width - box_width / 2
          - y_max = box_height - box_height / 2
        The main advantage of this format is that we can easily tranlate the 
        anchors by just adding the desired offset vector
    """
    k = len(aspect_ratios) * len(scales)
    scales = np.array(scales)
    aspect_ratios = np.array(aspect_ratios)

    heights = np.sqrt(np.tile(scales, len(aspect_ratios)) / 
                      np.repeat(aspect_ratios, len(scales)))
    widths = heights * np.repeat(aspect_ratios, len(scales))

    heights = heights.reshape(-1, 1)
    widths = widths.reshape(-1, 1)

    anchors = np.hstack([-widths / 2,
                         -heights / 2,
                         widths - widths / 2,
                         heights - heights / 2])
    anchors = anchors.astype('int32')
    return anchors


def tile_anchors(anchors: Tensor, 
                 image_shape: Tuple[int, int], 
                 stride: int = 1) -> Tensor:
    pass


def generate_anchors(
        image_shape: Tuple[int, int],
        aspect_ratios: Sequence[float] = (.5, 1., 2.),
        scales: Sequence[float] = (512**2, 256**2, 128**2)) -> Tensor:
    """
    Generate anchors located at the origin, and afterwards tiles the anchors
    all over the feature map shape. The procedure to create those anchors is
    described in the Faster RCNN paper.

    This method is equivalent to the composition of single_point_anchors and
    tile_anchors.

    Parameters
    ----------
    image_shape: Tuple[int, int]
        Tuple containing the image dimensions H x W
    aspect_rations: Sequence[float], default (.5, 1., 2.)
        Different aspect ratios for each anchor
    scales: Sequence[float], default (512**2, 256**2, 128**2)
        Different scales for each anchor.

    Returns
    -------
    Tensor of shape [N, 4]
        The resulting tensor containing all anchors for the given image size.
        The first tensor dimension is N which corresponds to 
        len(aspect_ratios) * len(scales) * H * W
        where H and W are the image height and width respectively
    """
    return single_point_anchors(aspect_ratios=aspect_ratios, scales=scales)
