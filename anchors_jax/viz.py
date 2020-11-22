
import itertools
from enum import Enum
from typing import Any, List, Union, Tuple

import numpy as onp

import jax
import jax.numpy as np

import PIL
from PIL import ImageDraw

from anchors_jax.typing import Box, Boxes, Image, Tensor


class Color(str, Enum):
    Red = '#ff0000'
    Green = '#00ff00'
    Blue = '#0000ff'
    Yellow = '#ffff00'
    Purple = '#ab47bc'
    White = '#ffffff'
    Black = '#000000'


def colored(items: List[Any]) -> List[Color]:
    """
    Get a color for each unique element of the list

    Examples
    --------
    >>> labels = ['cat', 'dog', 'dog', 'cat']
    >>> colored(labels)
    ... ['green', 'red, 'red, 'green']
    >>> labels = ['cat', 'dog', 'dog', 'cat']
    >>> colors = colored(labels)
    >>> draw_boxes(image, labels, colors=colors)
    """
    colors = [o.value for o in Color]
    unique_items = set(items)
    color_x_label = dict(zip(unique_items, itertools.cycle(colors)))
    return [color_x_label[o] for o in items]


def draw_boxes(
        im: Image, 
        boxes: Boxes, 
        labels: List[str] = None,
        colors: Union[List[Color], Color] = (255, 255, 255),
        boxes_width: int = 1,
        boxes_normalized: bool = True) -> PIL.Image:

    """
    Parameters
    ----------
    im: Image
        Image where to draw the boxes
    boxes: Boxes
        Boxes to draw, this can be a tensor or a usual python list
    colors: Union[List[RGBColor], RGBColor], default (255, 255, 255)
        colors parameter can be an iterable having the same length as boxes, 
        so each box is drew with the corresponding color or a single color to
        only use the specified one.
    boxes_width: int, default 1
        Outline with of the boxes
    boxes_normalized: bool, default True
        Are boxes scaled between 0 and 1?

    Returns
    -------
    PIL.Image
    """
    im = _parse_image(im)
    boxes = _parse_boxes(boxes, boxes_normalized, im.size)

    if labels is None:
        labels = [''] * len(boxes)

    if not isinstance(colors, list):
        colors = [colors]

    draw_im = im.copy()
    draw = ImageDraw.Draw(draw_im)

    for c, b, l in zip(itertools.cycle(colors), boxes, labels):

        if l != '':
            x1, y1, x2, y2 = b

            draw.rectangle([x1, y1 - int(im.size[1] * 0.06), 
                            x2, y1], fill=c,
                            outline=c, width=boxes_width)
            draw.text((x1 + 3, y1 - int(im.size[1] * 0.05)), 
                    l, fill=Color.Black)

        draw.rectangle(b, outline=c, width=boxes_width)

    return draw_im


def _parse_boxes(boxes: Boxes, 
                 boxes_normalized: bool, 
                 size: Tuple[int, int]) -> List[List[int]]:
    if isinstance(boxes, list):
        return [_parse_box(o, boxes_normalized, size) for o in boxes]

    elif isinstance(boxes, (onp.ndarray, onp.generic, np.ndarray, Tensor)):
        boxes = boxes.astype('float32').reshape(-1, 4)
        if boxes_normalized:
            x1, y1, x2, y2 = onp.split(boxes, 4, axis=-1)
            w, h = size
            x1 = x1 * w
            x2 = x2 * w
            y1 = y1 * h
            y2 = y2 * h
            boxes = onp.concatenate([x1, y1, x2, y2], axis=-1)
        return boxes.astype('int32').tolist()

    else:
        raise TypeError(f"Unexpected type {type(boxes)} for a "
                         "collection of boxes")


def _parse_box(box: Box, 
               boxes_normalized: bool, 
               size: Tuple[int, int]) -> List[int]:
    if isinstance(box, (onp.ndarray, onp.generic, np.ndarray)):
        box = box.astype('float32').reshape(-1)

        if boxes_normalized:
            x1, y1, x2, y2 = np.split(box, 4, axis=-1)
            w, h = size
            x1 = x1 * w
            x2 = x2 * w
            y1 = y1 * h
            y2 = y2 * h
            box = onp.concatenate([x1, y1, x2, y2], axis=-1)

        return box.tolist()

    elif isinstance(box, tuple):
        return list(box)

    elif isinstance(box, list):
        return box

    else:
        raise TypeError(f"Unexpected type {type(box)} for a box")


def _parse_image(im: Image) -> PIL.Image:
    """If necessary, converts a JAX or numpy tensor to a pillow image"""
    if isinstance(im, (onp.ndarray, onp.generic, jax.xla.DeviceArray)):
        # Jax arrays also enter here
        return PIL.Image.fromarray(onp.asarray(im).astype('uint8'))
    elif isinstance(im, PIL.Image.Image):
        return im
    else:
        raise TypeError(f"Unexpected type {type(im)} for an image")
