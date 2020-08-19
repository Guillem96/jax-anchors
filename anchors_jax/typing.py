from typing import List, Union, Tuple

import numpy as onp
import jax.numpy as np

from PIL import Image


Tensor = Union[onp.ndarray, np.ndarray]
Image = Union[Tensor, 'Image']

Box = Union[Tensor, List[int], Tuple[int, int, int, int]]
Boxes = Union[Tensor, List[Box]]

RGBColor = Tuple[int, int, int]