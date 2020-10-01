# -*- coding: utf-8 -*-

import jax.numpy as np

from anchors_jax.typing import Tensor


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
def iou(anchors: Tensor, boxes: Tensor) -> Tensor:
    """
    Computes Jaccard index between anchors and ground truth boxes.

    To vectorize this function so it is able to work with batches of boxes
    use the `jax.vmap` decorator as follows:

    `vmap_iou_fn = jax.vmap(iou, in_axes=[None, 0], out_axes=0)`

    Note that we do not batch anchors parameter, this is because anchors are equal
    along a batch

    Parameters
    ----------
    anchors: Tensor of shape [N, 4]
        Tensor containing the anchors for an specific image. anchors are boxes
        in the following format [x_min, y_min, x_max, y_max]
    boxes: Tensor of shape [M, 4]
        Tensor containing the boxes for the ground truths annotations. Ground
        truth boxes have the following format [x_min, y_min, x_max, y_max]
    
    Returns
    -------
    Tensor of shape [N, M] where each position corresponds to the pairwise iou
    value. For example, the index [10, 4] contains the iou magnitude representing
    the intersection between the 10th anchor and 4th box
    """
    anchors = anchors.astype('float32')
    boxes = boxes.astype('float32')

    a_x1, a_y1, a_x2, a_y2 = np.split(anchors, 4, axis=1)
    b_x1, b_y1, b_x2, b_y2 = np.split(boxes, 4, axis=1)

    anchors_areas = (a_y2 - a_y1) * (a_x2 - a_x1)
    boxes_areas = (b_y2 - b_y1) * (b_x2 - b_x1)

    lt = np.maximum(anchors.reshape(-1, 1, 4)[..., :2], boxes[..., :2])
    rb = np.minimum(anchors.reshape(-1, 1, 4)[..., 2:], boxes[..., 2:])
    wh = np.clip(rb - lt, 0)

    inter = wh[..., 0] * wh[..., 1]

    iou = inter / (anchors_areas + boxes_areas.reshape(-1) - inter)
    return iou.astype('float32')
