from enum import Enum
from typing import Any, Callable, List, Union, Tuple

import numpy as onp
import jax.numpy as np

from PIL import Image


Tensor = Union[onp.ndarray, np.ndarray]
Image = Union[Tensor, 'Image']

Box = Union[Tensor, List[int], Tuple[int, int, int, int]]
Boxes = Union[Tensor, List[Box]]

InitFn = Callable[[Tensor, Tuple[int, ...]], Tensor]
ForwardFn = Callable[[Tensor], Tensor]

Layer = Tuple[InitFn, ForwardFn]
LayerFactory = Callable[[Any], Layer]


class BoxesFormat(Enum):
    xyxy = 'xyxy'
    cxcywh = 'cxcywh'
    xywh = 'xywh'
