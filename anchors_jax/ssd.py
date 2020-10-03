from typing import Tuple, Sequence

import jax
import jax.numpy as np

import anchors_jax.ops as ops
import anchors_jax.boxes as utils
from anchors_jax.typing import Tensor


def scales_for_sizes(m: int, 
                     s_min: float = .2, 
                     s_max: float = .9) -> Sequence[float]:
    return [s_min + ((s_max - s_min) / (m - 1)) * (k - 1) 
            for k in range(1, m + 1)]


def single_point_anchors(
    aspect_ratios: Sequence[float] = (1., 2., 3., 1/2., 1/3.),
    scales: Sequence[float] = None) -> Sequence[Tensor]:

    """
    Generate K anchors for each scale. SSD works with different feature maps
    scales, therefore, K is equal to the number of different aspect ratios and
    the total generated anchors will be the number of the scales times the 
    number of aspect ratios

    Parameters
    ----------
    aspect_ratios: Sequence[float], default (1., 2., 3., 1/2., 1/3.)
    scales: Sequence[float], default None
        If left to None, by default the method computes the scale value for 
        three features maps
    
    Returns
    -------
    List[Tensor]
        Each element of the list contain K anchors (default boxes) and the list
        cardinality is equal to the length of the given scales
    """
    result = []
    aspect_ratios = np.array(aspect_ratios)
    scales = scales or scales_for_sizes(3)
    
    for i, sk in enumerate(scales):
        wa = sk * np.sqrt(aspect_ratios)
        wa = wa.reshape(-1, 1)
        
        ha = sk / np.sqrt(aspect_ratios)
        ha = ha.reshape(-1, 1)

        zeros = np.zeros((ha.shape[0], 1))
        anchors = np.hstack([zeros, zeros, wa, ha])

        if i < len(scales) - 2:
            hw = np.sqrt(sk * scales[i + 1])
            extra = np.array([0., 0., hw, hw])
            anchors = np.append(anchors, extra.reshape(1, 4), axis=0)

        result.append(anchors)

    return result


def tile_anchors(
        anchors: Sequence[Tensor], 
        feature_maps_shapes: Sequence[Tuple[int, int]]) -> Sequence[Tensor]:
    """
    Takes a sequence of anchors and tiles them to the corresponding feature map

    Parameters
    ----------
    anchors: Sequence[Tensor]
        Sequence of tensors of of shape [K, 4]. Usually this is sequence is the
        resulting from the `single_point_anchors` function. The anchors format 
        is [cx, cy, w, h].
    feature_maps_shapes: Sequence[Tuple[int, int]]
        Sequence of Tuples containing the feature maps dimensions (H x W) where the
        anchors are going to be tiled.
    
    Returns
    -------
    Sequence[Tensor], Tensors of shape [N, K, 4]
        A tensor of boxes of shape [N, K, 4] where N is the number of locations
        around the feature map (H x W / stride) and K is `anchors.shape[0]`. 
        Therefore, at each location we have K anchors.
    """
    def arange(limit):
        return (np.arange(limit).astype('float32') + .5) / limit

    assert len(anchors) == len(feature_maps_shapes), \
        "Feature maps shapes and anchors must have the same length"

    assert all([h == w for h, w in feature_maps_shapes]), \
        "Feature maps shapes have to be squared"

    tiled_anchors = []

    # We assume that feature maps are squared
    for fm_anchors, (fk, _) in zip(anchors, feature_maps_shapes):
        shifts_y, shifts_x = np.meshgrid(arange(fk), arange(fk))
        shifts_x = shifts_x.reshape(-1, 1)
        shifts_y = shifts_y.reshape(-1, 1)
        zeros = np.zeros_like(shifts_y)
        shifts = [shifts_x, shifts_y, zeros, zeros]
        shifts = np.concatenate(shifts, axis=1)

        N = shifts.shape[0]
        K = fm_anchors.shape[0]

        fm_tiled_anchors = fm_anchors.reshape(1, K, 4) + shifts.reshape(N, 1, 4)
        fm_tiled_anchors = fm_tiled_anchors.reshape(-1, 4)
        tiled_anchors.append(fm_tiled_anchors)

    return tiled_anchors


def generate_anchors(
        feature_maps_shapes: Sequence[Tuple[int, int]],
        aspect_ratios: Sequence[float] = (1., 2., 3., 1/2., 1/3.),
        s_min: float = .2, s_max: float = .9) -> Sequence[Tensor]:
    """
    Generate anchors located at the origin of each feature map, and afterwards 
    tiles them. This method generates the anchors as described in the SSD paper

    This method is equivalent to the composition of `single_point_anchors` and
    `tile_anchors`.

    Parameters
    ----------
    feature_maps_shapes: Sequence[Tuple[int, int]]
        Sequence of Tuples containing the feature maps dimensions (H x W) where the
        anchors are going to be tiled.
    aspect_ratios: Sequence[float], default (1., 2., 3., 1/2., 1/3.)
    s_min: float, default .2
        Scale for the largest feature map
    s_max: float, default .9
        Scale for the smallest feature map. As smaller is the feature map the 
        larger the scale will be. This is because smaller feature maps have
        a larger receptive field.

    Returns
    -------
    Sequence[Tensor], Tensors of shape [N, 4]
        The resulting tensor containing all anchors for every feature map.
        The first tensor dimension is N which corresponds to 
        len(aspect_ratios) * len(scales) * H * W
        where H and W are the image height and width respectively
    """
    m = len(feature_maps_shapes)
    scales = scales_for_sizes(m, s_min, s_max)
    anchors = single_point_anchors(aspect_ratios=aspect_ratios, scales=scales)
    return tile_anchors(anchors, feature_maps_shapes=feature_maps_shapes)
