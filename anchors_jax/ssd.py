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
        The resulting anchors are formated as [cx, cy, w, h]. Anchors are 
        normalized between 0 to 1 with respect to the feature map shape
    """
    m = len(feature_maps_shapes)
    scales = scales_for_sizes(m, s_min, s_max)
    anchors = single_point_anchors(aspect_ratios=aspect_ratios, scales=scales)
    return tile_anchors(anchors, feature_maps_shapes=feature_maps_shapes)


def detect_tag_anchors(anchors: Sequence[Tensor], 
                       boxes: Tensor, 
                       labels: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Tags every anchor with the corresponding classification and regression labels

    Note that the labels are numberic and start with 1, the 0 value is reserved
    by the background or negative anchors.

    Parameters
    ----------
    anchors: Sequence of Tensors of shape [N, 4]
        Anchors are expected to be normalized between 0 and 1 and formated as
        [cx, cy, w, h].
    boxes: Tensor of shape [M, 4]
        Boxes are expected to be normalized between 0 and 1 with respect to the
        original image size and formated as [x_min, y_min, x_max, y_max]
    labels: Tensor of shape [M] or [M, 1]

    Returns
    -------
    Sequence[Tuple[Tensor, Tensor]]
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

    ks = [0] + [len(o) for o in anchors]
    ks = np.cumsum(np.array(ks))

    anchors = np.concatenate(anchors, axis=0)
    anchors = utils.cxcywh_to_xyxy(anchors)

    anchors_indices = _anchors_indices(anchors, boxes)
    positive_mask, negative_mask, selected_boxes_idx = anchors_indices

    selected_boxes = boxes[selected_boxes_idx]
    selected_labels = labels[selected_boxes_idx].reshape(-1).astype('float32')

    # Non used labels are ignored with -1
    cls_labels = np.zeros((anchors.shape[0], )) - 1
    cls_labels = np.where(negative_mask, 0., cls_labels)
    cls_labels = np.where(positive_mask, selected_labels, cls_labels)
    cls_labels = cls_labels.reshape(-1, 1)

    # Start with the regressors
    regressors = _compute_regressors(anchors, selected_boxes)

    # Only keep positive anchor regressors, override the negative and ignored
    # ones with zeros
    keep_regressors = positive_mask.astype('float32')
    keep_regressors = np.repeat(np.expand_dims(keep_regressors, -1), 4, axis=-1)
    regressors = regressors * keep_regressors

    cls_labels, regressors = zip(*[(cls_labels[k1:k2], regressors[k1:k2]) 
                                   for k1, k2 in zip(ks[:-1], ks[1:])])

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
    negative_mask = highest_boxes_iou < .5

    return positive_mask, negative_mask, highest_boxes_iou_idx


def _compute_regressors(anchors: Tensor, boxes: Tensor) -> Tensor:
    assert anchors.shape[0] == boxes.shape[0]

    anchors = utils.xyxy_to_cxcywh(anchors)
    boxes = utils.xyxy_to_cxcywh(boxes)

    x_a, y_a, w_a, h_a = np.split(anchors, 4, axis=1)
    x_star, y_star, w_star, h_star = np.split(boxes, 4, axis=1)

    # Regressors 
    tx_star = (x_star - x_a) / w_a
    ty_star = (y_star - y_a) / h_a
    tw_star = np.where(w_star > 0., np.log(w_star / w_a), 0.)
    th_star = np.where(h_star > 0., np.log(h_star / h_a), 0.)

    return np.concatenate([tx_star, ty_star, tw_star, th_star], axis=-1)
