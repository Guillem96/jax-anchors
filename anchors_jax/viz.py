from typing import List, Union

import numpy as onp
import jax.numpy as np

import PIL

from anchors_jax.typing import Box, Boxes, Image, RGBColor


def draw_boxes(
        im: Image, 
        boxes: Boxes, 
        colors: Union[List[RGBColor], RGBColor] = (255, 255, 255)) -> PIL.Image:

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
    
    Returns
    -------
    PIL.Image
    """
    im = _parse_image(im)
    boxes = _parse_boxes(boxes)
    if not isinstance(colors, list):
        colors = [colors] * len(boxes)
    
    draw_im = im.copy()
    draw = PIL.ImageDraw.Draw(draw_im)

    for c, b in zip(colors, boxes):
        draw.rectangle(b, outline=c)

    return draw_im


def _parse_boxes(boxes: Boxes) -> List[List[int]]:
    if isinstance(boxes, list):
        return [_parse_box(o) for o in boxes]
    elif isinstance(boxes, (onp.ndarray, onp.generic, np.ndarray)):
        return boxes.astype('int32').reshape(-1, 4).tolist()
    else:
        raise TypeError(f"Unexpected type {type(boxes)} for a "
                         "collection of boxes")


def _parse_box(box: Box) -> List[int]:
    if isinstance(box, (onp.ndarray, onp.generic)):
        return box.astype('int32').reshape(-1).tolist()
    elif isinstance(box, tuple):
        return list(box)
    else:
        raise TypeError(f"Unexpected type {type(box)} for a box")


def _parse_image(im: Image) -> PIL.Image:
    """If necessary, converts a JAX or numpy tensor to a pillow image"""
    if isinstance(im, (onp.ndarray, onp.generic)):
        # Jax arrays also enter here
        return PIL.Image.fromarray(onp.asarray(im).astype('uint8'))
    elif isinstance(im, PIL.Image.Image):
        return im
    else:
        raise TypeError(f"Unexpected type {type(im)} for an image")
