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


        anchors = np.hstack([-wa, -ha, wa, ha]) / 2.
        if i < len(scales) - 2:
            hw = np.sqrt(sk * scales[i + 1])
            extra = np.array([-hw, -hw, hw, hw]) / 2.
            anchors = np.append(anchors, extra.reshape(1, 4), axis=0)

        result.append(anchors)

    return result
