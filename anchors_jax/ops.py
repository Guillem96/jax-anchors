# -*- coding: utf-8 -*-

import jax
import jax.numpy as np

from anchors_jax.typing import Tensor, BoxesFormat
import anchors_jax.boxes as boxes_utils

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

    iou = inter / (anchors_areas + boxes_areas.reshape(-1) - inter + 1e-8)

    return iou.astype('float32')


def class_wise_nms(boxes: Tensor, 
                   scores: Tensor,
                   classes: Tensor,
                   overlap_threshold: float = .5,
                   score_threshold: float = .5,
                   boxes_fmt: BoxesFormat = BoxesFormat.xyxy) -> Tensor:
    
    """
    Performs Non Maxima Supperssion for each unique class with the given boxes 
    Selects the boxes with higher score and discards the ones pointing to the
    same object.

    Parameters
    ----------
    boxes: Tensor of shape [N, 4]
        Boxes formated according to the input parameter `boxes_fmt`
    scores: Tensor of shape [N]
        Boxes scores ranging from 0 to 1 being 1 a higher value
    classes: Tensor of shape [N]
        Boxes classes
    boxes_fmt: BoxesFormat, default xyxy
        Format of the boxes, by default it is set to 
        [x_min, y_min, x_max, y_max]
    overlap_threshold: float, default .5
        Overlapping boxes pointing to the same object with an iou larger than
        this threshold are going to be discarded. NMS will only keep the one
        with the highest score
    score_threshold: float, default 0.5
        Boxes with a lower score than score_threshold will be discarded
    """

    if boxes_fmt != BoxesFormat.xyxy:
        convert_fn = getattr(boxes_utils, f'{boxes_fmt.value}_to_xyxy')
        boxes = convert_fn(boxes)

    masks = np.zeros(boxes.shape[0], dtype='bool')
    classes = classes.reshape(-1).astype('int32')
    n_classes = np.max(classes)

    # Per class NMS
    # TODO: Should labels always start at 1?
    for c in np.arange(1, n_classes + 1):
        if c == -1:
            continue
        
        mask = (classes == c).reshape(-1)
        current_scores = np.where(mask, scores, 0.)
        current_boxes = np.where(np.expand_dims(mask, -1), boxes, 0.)

        boxes_mask = nms(boxes=current_boxes,
                         scores=current_scores,
                         overlap_threshold=overlap_threshold,
                         score_threshold=score_threshold)

        masks = masks | (boxes_mask & mask)

    return masks


def nms(boxes: Tensor, 
        scores: Tensor,
        overlap_threshold: float = .5,
        score_threshold: float = .5,
        boxes_fmt: BoxesFormat = BoxesFormat.xyxy) -> Tensor:
    
    """
    Performs Non maxima supperssion with the given boxes
    Selects the boxes with higher score and discards the ones pointing to the
    same object.

    Parameters
    ----------
    boxes: Tensor of shape [N, 4]
        Boxes formated according to the input parameter `boxes_fmt`
    scores: Tensor of shape [N]
        Boxes scores ranging from 0 to 1 being 1 a higher value
    boxes_fmt: BoxesFormat, default xyxy
        Format of the boxes, by default it is set to 
        [x_min, y_min, x_max, y_max]
    overlap_threshold: float, default .5
        Overlapping boxes pointing to the same object with an iou larger than
        this threshold are going to be discarded. NMS will only keep the one
        with the highest score
    score_threshold: float, default 0.5
        Boxes with a lower score than score_threshold will be discarded

    Returns
    -------
    Tensor of shape [N]
        A mask containing True if the box has to be kept and False otherwise
    
    Examples
    --------
    >>> N = 10
    >>> boxes = jax.random.uniform(key, shape=(N, 4))
    >>> scores = np.ones(N)
    >>> keep_mask = aj.ops.nms(boxes, scores)
    >>> desired_boxes = boxes[keep_mask]

    """

    if boxes_fmt != BoxesFormat.xyxy:
        convert_fn = getattr(boxes_utils, f'{boxes_fmt.value}_to_xyxy')
        boxes = convert_fn(boxes)

    n = boxes.shape[0]

    # Sort boxes by score
    sort_idx = np.argsort(-scores)
    scores = np.take(scores, sort_idx)
    boxes = boxes[sort_idx]

    # Compute iou of sorted boxes, hence the boxes with lower scores are going
    # to be at the end of the rows
    ious = iou(boxes, boxes)

    # Set itsself iou to 0 (iou matrix diag to 0)
    ious_mask = 1 - np.eye(n)
    ious = ious * ious_mask

    score_keep_mask = scores > score_threshold

    overlapping_boxes = ious > overlap_threshold

    # Since boxes with lower scores are at the end of the iou matrix rows
    # we create a mask containing ones for every element which is over the
    # diagonal. For box at index 3 all elements after the third index of its
    # row are going to have a lower score
    smaller_score_than_mask = sum([np.eye(n, k=i) for i in range(n)])
    smaller_score_than_mask = smaller_score_than_mask.astype('bool')

    # If a box overlaps and has a lower score we discard it
    # To compute the keep mask we invert the discard mask
    iou_keep_mask = ~(overlapping_boxes & smaller_score_than_mask)
    iou_keep_mask = np.all(iou_keep_mask, axis=0)

    keep_mask = iou_keep_mask & score_keep_mask

    # Undo the sort and return the mask according to the input boxes
    init_correspondence_idx = jax.ops.index_update(
        np.zeros_like(sort_idx),
        sort_idx,
        np.arange(sort_idx.shape[0]))
 
    return keep_mask[init_correspondence_idx]
