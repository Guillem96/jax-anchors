from typing import Tuple

import jax.numpy as np

from anchors_jax.typing import Tensor, BoxesFormat


def area(boxes: Tensor, 
         boxes_fmt: BoxesFormat = BoxesFormat.xyxy) -> Tensor:

    if boxes_fmt != BoxesFormat.cxcywh:
        convert_fn = getattr('.', f'{boxes_fmt.value}_to_cxcywh')
        boxes = convert_fn(boxes)

    return boxes[..., 2] * boxes[..., 3]


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts a tensor of boxes formatted as [x_center, y_center, width, height]
    to [x_min, y_min, x_max, y_max]

    Parameters
    ----------
    boxes: Tensor of shape [N, 4]
        Tensor of N boxes formatted as [x_center, y_center, width, height]
    
    Returns
    -------
    Tensor of shape [N, 4]
        Tensor of N boxes formatted as [x_min, y_min, x_max, y_max]
    """
    x, y, w, h = np.split(boxes, 4, axis=1)

    x_min = x - w / 2.
    y_min = y - h / 2.
    x_max = x_min + w
    y_max = y_min + h
    return np.concatenate([x_min, y_min, x_max, y_max], axis=-1)


def xyxy_to_xywh(boxes) :
    """
    Converts a tensor of boxes formatted as [x_min, y_min, x_max, y_max] to
    [x_min, y_min, width, height]

    Parameters
    ----------
    boxes: Tensor of shape [N, 4]
        Tensor of N boxes formatted as [x_min, y_min, x_max, y_max]
    
    Returns
    -------
    Tensor of shape [N, 4]
        Tensor of N boxes formatted as [x_min, y_min, width, height]
    """
    x_min, y_min, x_max, y_max = np.split(boxes, 4, axis=1)

    w = x_max - x_min
    h = y_max - y_min

    return np.concatenate([x_min, y_min, w, h], axis=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """
    Converts a tensor of boxes formatted as [x_min, y_min, x_max, y_max] to
    [x_center, y_center, width, height]

    Parameters
    ----------
    boxes: Tensor of shape [N, 4]
        Tensor of N boxes formatted as [x_min, y_min, x_max, y_max]
    
    Returns
    -------
    Tensor of shape [N, 4]
        Tensor of N boxes formatted as [x_center, y_center, width, height]
    """
    x_min, y_min, x_max, y_max = np.split(boxes, 4, axis=1)

    x = (x_min + x_max) / 2.
    y = (y_min + y_max) / 2.
    w = x_max - x_min
    h = y_max - y_min

    return np.concatenate([x, y, w, h], axis=-1)


def normalize_boxes(boxes: Tensor, im_shape: Tuple[int, int]) -> Tensor:
    h, w = im_shape

    x_min, y_min, x_max, y_max = np.split(boxes, 4, axis=1)

    x_min = x_min / w
    y_min = y_min / h
    x_max = x_max / w
    y_max = y_max / h

    return np.concatenate([x_min, y_min, x_max, y_max], axis=1)


def scale_boxes(boxes: Tensor, im_shape: Tuple[int, int]) -> Tensor:
    h, w = im_shape

    x_min, y_min, x_max, y_max = np.split(boxes, 4, axis=1)

    x_min = x_min * w
    y_min = y_min * h 
    x_max = x_max * w 
    y_max = y_max * h

    return np.concatenate([x_min, y_min, x_max, y_max], axis=1)