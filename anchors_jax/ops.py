# -*- coding: utf-8 -*-

import jax.numpy as np

from anchors_jax.typing import Tensor


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

    total_areas = anchors_areas + boxes_areas.T

    inter_x_min = np.maximum(a_x1, b_x1.T)
    inter_y_min = np.maximum(a_y1, b_y1.T)
    inter_x_max = np.minimum(a_x2, b_x2.T)
    inter_y_max = np.minimum(a_y2, b_y2.T)

    inter_w = inter_x_max - inter_x_min
    inter_w = np.maximum(inter_w, 0)
    inter_h = inter_y_max - inter_y_min
    inter_h = np.maximum(inter_h, 0)
    inter_areas = inter_h * inter_w

    return inter_areas / (total_areas - inter_areas + 1e-8).astype('float32')
