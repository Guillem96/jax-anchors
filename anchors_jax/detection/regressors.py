import jax.numpy as np

from anchors_jax.typing import Tensor, BoxesFormat
from anchors_jax import boxes as boxes_utils


def compute_regressors(anchors: Tensor, boxes: Tensor,
                       anchors_fmt: BoxesFormat = BoxesFormat.cxcywh,
                       boxes_fmt: BoxesFormat = BoxesFormat.cxcywh,
                       standardize: bool = True) -> Tensor:
    assert anchors.shape[0] == boxes.shape[0]

    # Convert anchors and boxes to cxcywh
    if anchors_fmt != BoxesFormat.cxcywh:
        convert_fn = getattr(boxes_utils, f'{anchors_fmt.value}_to_cxcywh')
        anchors = convert_fn(anchors)

    if boxes_fmt != BoxesFormat.cxcywh:
        convert_fn = getattr(boxes_utils, f'{boxes_fmt.value}_to_cxcywh')
        boxes = convert_fn(boxes)

    x_a, y_a, w_a, h_a = np.split(anchors, 4, axis=1)
    x_star, y_star, w_star, h_star = np.split(boxes, 4, axis=1)

    # Regressors 
    tx_star = np.where(w_a <= 0., 0., (x_star - x_a) / w_a)
    ty_star = np.where(h_a <= 0., 0., (y_star - y_a) / h_a)
    tw_star = np.where((w_star > 0.) & (w_a > 0.), np.log(w_star / w_a), 0.)
    th_star = np.where((h_star > 0.) & (h_a > 0.), np.log(h_star / h_a), 0.)

    if standardize:
        mean = [0., 0., 0., 0.]
        std = [0.2, 0.2, 0.2, 0.2]

        tx_star = (tx_star - mean[0]) / std[0]
        ty_star = (ty_star - mean[1]) / std[1]
        tw_star = (tw_star - mean[2]) / std[2]
        th_star = (th_star - mean[3]) / std[3]

    return np.concatenate([tx_star, ty_star, tw_star, th_star], axis=-1)


def apply_regressors(anchors: Tensor,
                     regressors: Tensor,
                     anchors_fmt: BoxesFormat = BoxesFormat.cxcywh,
                     standardize: bool = True) -> Tensor:
    """
    Corrects the anchors with the predicted regressors

    Parameters
    ----------
    anchors: Tensor of shape [N, 4]
        Anchors are expected to be normalized between 0 and 1 and formated as
        [cx, cy, w, h].
    regressors: Tensor of shape [N, 4]
    standardize: bool, default True
        User should set this to true in case the regressors are standardized

    Returns
    -------
    Tensor of shape [N, 4]
    """
    assert anchors.shape[0] == regressors.shape[0]

    # Convert anchors and boxes to cxcywh
    if anchors_fmt != BoxesFormat.cxcywh:
        convert_fn = getattr(boxes_utils, f'{anchors_fmt.value}_to_cxcywh')
        anchors = convert_fn(anchors)

    x_a, y_a, w_a, h_a = np.split(anchors, 4, axis=1)
    tx, ty, tw, th = np.split(regressors, 4, axis=1)

    if standardize:
        mean = [0., 0., 0., 0.]
        std = [0.2, 0.2, 0.2, 0.2]

        tx = tx * std[0] + mean[0]
        ty = ty * std[1] + mean[1]
        tw = tw * std[2] + mean[2]
        th = th * std[3] + mean[3]

    x = tx * w_a + x_a
    y = ty * h_a + y_a
    w = np.clip(w_a * np.exp(tw), a_max=10.)
    h = np.clip(h_a * np.exp(th), a_max=10.)

    return np.concatenate([x, y, w, h], axis=-1)
