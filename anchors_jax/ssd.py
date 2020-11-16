import itertools
from typing import List, Tuple, Sequence, Union

import jax
import jax.numpy as np

import anchors_jax.ops as ops
import anchors_jax.boxes as boxes_utils
from anchors_jax.typing import BoxesFormat, Tensor
from anchors_jax.detection.regressors import (apply_regressors, 
                                              compute_regressors)


AspectRatios = Union[Tuple[float, ...], List[Tuple[float, ...]]]


def scales_for_sizes(m: int, 
                     s_min: float = .2, 
                     s_max: float = .9) -> Sequence[float]:
    return [s_min + ((s_max - s_min) / (m - 1)) * (k - 1) 
            for k in range(1, m + 2)]


def single_point_anchors(
    aspect_ratios: AspectRatios = (1., 2., 3., 1/2., 1/3.),
    scales: Sequence[float] = None) -> Sequence[Tensor]:

    """
    Generate K anchors for each scale. SSD works with different feature maps
    scales, therefore, K is equal to the number of different aspect ratios and
    the total generated anchors will be the number of the scales times the 
    number of aspect ratios

    Parameters
    ----------
    aspect_ratios: Tuple[float, ...] or List[Tuple[float, ...]], default (1., 2., 3., 1/2., 1/3.)
        Aspect ratios per level
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
    scales = scales or scales_for_sizes(3)

    if not isinstance(aspect_ratios, list):
        aspect_ratios = itertools.cycle([aspect_ratios])

    elif len(aspect_ratios) != len(scales) - 1:
        raise ValueError("The length of aspect ratio and scales must be equal")

    for i, (sk, ar) in enumerate(zip(scales[:-1], aspect_ratios)):
        ar = np.array(ar)
        wa = sk * np.sqrt(ar)
        wa = wa.reshape(-1, 1)
        
        ha = sk / np.sqrt(ar)
        ha = ha.reshape(-1, 1)

        zeros = np.zeros((ha.shape[0], 1))
        anchors = np.hstack([zeros, zeros, wa, ha])

        hw = np.sqrt(sk * scales[i + 1])
        extra = np.array([0., 0., hw, hw])
        anchors = np.append(anchors, extra.reshape(1, 4), axis=0)

        result.append(anchors)

    return result


def tile_anchors(
        anchors: Sequence[Tensor], 
        feature_maps_shapes: Sequence[Tuple[int, int]]) -> Tensor:
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
    Tensor of shape [N, 4]
        A tensor of boxes of shape [N, 4] where N is the number of locations
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
        shifts_x, shifts_y = np.meshgrid(arange(fk), arange(fk))
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

    return np.concatenate(tiled_anchors)


def generate_anchors(
        feature_maps_shapes: Sequence[Tuple[int, int]],
        aspect_ratios: Sequence[float] = (1., 2., 3., 1/2., 1/3.),
        s_min: float = .2, s_max: float = .9) -> Tensor:
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
    Tensor of shape [N, 4]
        The resulting tensor containing all anchors for every feature map.
        The resulting anchors are formated as [cx, cy, w, h]. Anchors are 
        normalized between 0 to 1 with respect to the feature map shape
    """
    m = len(feature_maps_shapes)
    scales = scales_for_sizes(m, s_min, s_max)
    anchors = single_point_anchors(aspect_ratios=aspect_ratios, scales=scales)
    return tile_anchors(anchors, feature_maps_shapes=feature_maps_shapes)


def detect_tag_anchors(
        anchors: Tensor, 
        boxes: Tensor, 
        labels: Tensor,
        standardize_regressors: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Tags every anchor with the corresponding classification and regression labels

    Note that the labels are numberic and start with 1, the 0 value is reserved
    by the background or negative anchors.

    Parameters
    ----------
    anchors: Tensor of shape [N, 4]
        Anchors are expected to be normalized between 0 and 1 and formated as
        [cx, cy, w, h].
    boxes: Tensor of shape [M, 4]
        Boxes are expected to be normalized between 0 and 1 with respect to the
        original image size and formated as [x_min, y_min, x_max, y_max]
    labels: Tensor of shape [M] or [M, 1]
    standardize_regressors: bool, default True
        Wether or not to standardize the regressors with 0 mean and .2 std

    Returns
    -------
    Tuple[Tensor, Tensor]
        First element of the tuple has shape [N, 1] and contains 0 if the anchor
        does not overlap with any box, the overlaping box label if the anchor 
        overlaps with a box or a -1 if the anchor has to be ignored.

        The second element of the tuple is a Tensor of shape [N, 4] containing the
        regressors for each anchor. The regressors are computed with the formulas
        specified in FasterRCNN paper. Note that the regressors on non-overlaping
        anchors should be filtered out. The regressors of ignored and negative
        anchors are -1.
    """
    assert boxes.shape[0] == labels.reshape(-1).shape[0]

    anchors = boxes_utils.cxcywh_to_xyxy(anchors)

    anchors_indices = _anchors_indices(anchors, boxes)
    positive_mask, negative_mask, selected_boxes_idx = anchors_indices

    selected_boxes = boxes[selected_boxes_idx]
    selected_labels = labels[selected_boxes_idx].reshape(-1).astype('float32')

    # Non used labels are ignored with -1
    cls_labels = np.zeros((anchors.shape[0], )) - 1
    cls_labels = np.where(negative_mask, 0., cls_labels)
    cls_labels = np.where(positive_mask, selected_labels, cls_labels)
    cls_labels = np.where(selected_labels == -1, -1, cls_labels) # Padding labels
    cls_labels = cls_labels.reshape(-1, 1)

    # Start with the regressors
    regressors = compute_regressors(anchors,
                                    selected_boxes,
                                    anchors_fmt=BoxesFormat.xyxy,
                                    boxes_fmt=BoxesFormat.xyxy,
                                    standardize=standardize_regressors)

    # Only keep positive anchor regressors, override the negative and ignored
    # ones with zeros
    keep_regressors = positive_mask.astype('float32')
    keep_regressors = np.repeat(np.expand_dims(keep_regressors, -1), 4, axis=-1)
    regressors = regressors * keep_regressors

    return cls_labels, regressors


def _anchors_indices(anchors: Tensor, 
                     boxes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

    ious = ops.iou(anchors, boxes)

    cx = (anchors[..., 2] + anchors[..., 0]) / 2.
    cy = (anchors[..., 1] + anchors[..., 3]) / 2.
    cross_boundary_x = (cx < 0.) | (cx > 1.)
    cross_boundary_y = (cy < 0.) | (cy > 1.)
    cross_boundary  = cross_boundary_x | cross_boundary_y

    highest_boxes_iou_idx = ious.argmax(-1)
    highest_boxes_iou = ious.max(-1)

    positive_mask = highest_boxes_iou >= .5
    positive_mask = positive_mask & ~cross_boundary

    negative_mask = highest_boxes_iou < .5
    negative_mask = negative_mask & ~cross_boundary

    return positive_mask, negative_mask, highest_boxes_iou_idx
