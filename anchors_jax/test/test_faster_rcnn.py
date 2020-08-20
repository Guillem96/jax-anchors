import unittest

import numpy as onp
from PIL import Image

import anchors_jax.faster_rcnn as aj
import anchors_jax.viz as viz


class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        im_arr = onp.random.uniform(size=(512, 512, 3)) * 255
        im_arr = im_arr.astype('uint8')

        self.im = Image.fromarray(im_arr)
        self.scales = 32**2, 64**2, 128**2

    def test_single_location_anchors(self):
        anchors = aj.single_point_anchors(scales=self.scales)
        self.assertEqual((9, 4), anchors.shape)
        center = self.im.size[0] // 2
        viz.draw_boxes(self.im, anchors + center, boxes_width=2).show()

    def test_tile_anchors(self):
        tiled_anchors = aj.generate_anchors(self.im.size[::-1], 
                                            scales=self.scales,
                                            stride=self.im.size[0] // 2 - 1)
        self.assertEqual(tiled_anchors.shape[1], 9)

        viz.draw_boxes(self.im, tiled_anchors.reshape(-1, 4), 
                       boxes_width=1).show()
