from typing import Tuple, Sequence

import jax
import jax.numpy as np

import anchors_jax.ops as ops
import anchors_jax.boxes as utils
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
    """
    Takes a set of anchors located at the origin, and tiles them over a feature
    map

    Parameters
    ----------
    anchors: Tensor of shape [K, 4]
        Set of starting anchors, usually generated with the function 
        `single_point_anchors`. Anchors are boxes with the following format
        [x_min, y_min, x_max, y_max]
    image_shape: Tuple[int, int]
        Tuple containing the image dimensions H x W
    stride: int, default 1
        Distance in pixels between the anchors centers
    
    Returns
    -------
    Tensor of shape [N, K, 4]
        A tensor of boxes of shape [N, K, 4] where N is the number of locations
        around the feature map (H x W / stride) and K is `anchors.shape[0]`. 
        Therefore, at each location we have K anchors.
    """
    K = anchors.shape[0]

    shifts_x, shifts_y = np.meshgrid(np.arange(0.5, image_shape[1], stride),
                                     np.arange(0.5, image_shape[0], stride))
    shifts_x = shifts_x.reshape(-1, 1)
    shifts_y = shifts_y.reshape(-1, 1)

    shifts = np.concatenate([shifts_x, shifts_y, shifts_x, shifts_y], axis=1)
    N = shifts.shape[0]
    tiled_anchors = anchors.reshape(1, K, 4) + shifts.reshape(N, 1, 4)
    return tiled_anchors


def generate_anchors(
        image_shape: Tuple[int, int],
        aspect_ratios: Sequence[float] = (.5, 1., 2.),
        scales: Sequence[float] = (512**2, 256**2, 128**2),
        stride: int = 1) -> Tensor:
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
    stride: int, default 1
        Distance in pixels between the anchors centers
    Returns
    -------
    Tensor of shape [N, 4]
        The resulting tensor containing all anchors for the given image size.
        The first tensor dimension is N which corresponds to 
        len(aspect_ratios) * len(scales) * H * W
        where H and W are the image height and width respectively
    """
    anchors = single_point_anchors(aspect_ratios=aspect_ratios, scales=scales)
    return tile_anchors(anchors, image_shape, stride=stride)


def rpn_tag_anchors(anchors: Tensor, boxes: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Assigns a classification label and a regressor to each anchor box.

    Parameters
    ----------
    anchors: Tensor of shape [N, 4]
    boxes: Tensor of shape [M, 4]
        To work with batches of boxes, use `jax.vmap` decorator as follows:
        `fn = jax.vmap(rpn_tag_anchors, in_axes=[None, 0])`, note that we do not
        batch anchors, this is because anchors are always equal along the batch
    
    Returns
    -------
    Tuple[Tensor, Tensor]
        First element of the tuple has shape [N, 1] and contains 0 if the anchor
        does not overlap with any box, 1 if the anchor overlaps with a box or a 
        -1 if the anchor has to be ignored.
        This objectness score is computed as specified in FasterRCNN paper:
        > We assign a positive label to two kinds of anchors: (i) the anchor/anchors 
          with the highest Intersection-overUnion (IoU) overlap with a 
          ground-truth box, or (ii) an anchor that has an IoU overlap higher than 0.7
        The second element of the tuple is a Tensor of shape [N, 4] containing the
        regressors for each anchor. The regressors are computed with the formulas
        specified in FasterRCNN paper. Note that the regressors on non-overlaping
        anchors should be filtered out. The regressors of ignored and negative
        anchors contains -1 components.
    """
    ious = ops.iou(anchors, boxes)

    # Compute positive mask:
    # - Either anchors with highest overlap with a box, and anchors with an iou 
    #   larger or equal than 0.7
    larger_ious = ious >= .7

    higher_than_07 = np.any(larger_ious, axis=-1)
    higher_than_07 = higher_than_07.astype('float32')

    highest_ious_anchors_idx = ious.T.argmax(-1)
    highest_ious_anchors = np.zeros((anchors.shape[0]))
    highest_ious_anchors = jax.ops.index_update(highest_ious_anchors,
                                                highest_ious_anchors_idx, 1.)

    positive_mask = higher_than_07 + highest_ious_anchors
    positive_mask = positive_mask.reshape(-1).astype('bool')

    # Compute negative mask, anchors with less than 0.3 iou
    negative_mask = np.all(ious <= .3, axis=-1).reshape(-1)

    # Non used labels are ignored with -1
    cls_labels = np.zeros((anchors.shape[0], )) - 1 
    cls_labels = np.where(positive_mask, 1., cls_labels)
    cls_labels = np.where(negative_mask, 0., cls_labels).reshape(-1, 1)

    # Start with the regressors
    selected_boxes_idx = larger_ious.argmax(-1)
    selected_boxes_idx = jax.ops.index_update(
        selected_boxes_idx, 
        highest_ious_anchors_idx, # highest_ious_anchors_idx, Contains the max 
                                  # overlaping anchor idx for each box
        np.arange(highest_ious_anchors_idx.shape[0]))

    selected_boxes = boxes[selected_boxes_idx]
    regressors = _compute_regressors(anchors, selected_boxes)
    
    # Remove negative and ignored anchors regressors
    regress_ignore = negative_mask | (cls_labels.reshape(-1) == -1.)
    regress_ignore_idx = regress_ignore.nonzero()
    regressors = jax.ops.index_update(regressors, regress_ignore_idx, 0.)

    return cls_labels, regressors


def apply_regressors(anchors: Tensor, regressors: Tensor) -> Tensor:
    """
    Corrects the anchors with the predicted regressors

    Parameters
    ----------
    anchors: Tensor of shape [N, 4]
    regressors: Tensor of shape [N, 4]

    Returns
    -------
    Tensor of shape [N, 4]
    """
    assert anchors.shape[0] == regressors.shape[0]
    
    anchors = utils.xyxy_to_cxcywh(anchors)

    x_a, y_a, w_a, h_a = np.split(anchors, 4, axis=1)
    tx, ty, tw, th = np.split(regressors, 4, axis=1)

    x = tx * w_a + x_a
    y = ty * h_a + y_a
    w = w_a * np.exp(tw)
    h = h_a * np.exp(th)

    return utils.cxcywh_to_xyxy(np.concatenate([x, y, w, h], axis=-1))


def _compute_regressors(anchors: Tensor, boxes: Tensor) -> Tensor:
    assert anchors.shape[0] == boxes.shape[0]

    anchors = utils.xyxy_to_cxcywh(anchors)
    boxes = utils.xyxy_to_cxcywh(boxes)

    x_a, y_a, w_a, h_a = np.split(anchors, 4, axis=1)
    x_star, y_star, w_star, h_star = np.split(boxes, 4, axis=1)

    # Regressors 
    tx_star = (x_star - x_a) / w_a
    ty_star = (y_star - y_a) / h_a
    tw_star = np.log(w_star / w_a)
    th_star = np.log(h_star / h_a)

    return np.concatenate([tx_star, ty_star, tw_star, th_star], axis=-1)
