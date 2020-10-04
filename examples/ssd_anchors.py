# -*- coding: utf-8 -*-

import sys
import json
from typing import List, Mapping
sys.path.append('..')

import click

from PIL import Image

import jax.numpy as np

import anchors_jax as aj


@click.command()
@click.argument('im_path')
@click.argument('annot_path')
@click.option('--save-path', type=str, default=None, 
              help="If set, the image will be stored in the specified path")
def main(im_path: str, annot_path: str, save_path: str):
    im = Image.open(im_path)
    im = im.resize((720, 720))

    # Load labelme annoations
    boxes, labels_idx, label_2_idx = aj.io.load_labelme_annot(annot_path)

    print("Generating anchors...")
    feature_maps_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    anchors = aj.ssd.generate_anchors(feature_maps_shapes=feature_maps_shapes)

    cls_labels, regressors = aj.ssd.detect_tag_anchors(
        anchors=anchors, boxes=boxes, labels=labels_idx)

    cls_labels = np.concatenate(cls_labels)
    regressors = np.concatenate(regressors)

    anchors = [aj.boxes.cxcywh_to_xyxy(o) for o in anchors]
    anchors = [aj.boxes.scale_boxes(o, im.size[::-1]) for o in anchors]
    anchors = np.concatenate(anchors)

    boxes = aj.boxes.scale_boxes(boxes, im.size[::-1])

    print('Drawing positive anchors...')
    im = aj.viz.draw_boxes(im, anchors[cls_labels.reshape(-1) > 0.],
                           boxes_width=2, 
                           colors=[aj.viz.Color.Red,])
    im.show()
    if save_path is not None:
        im.save(save_path)


if __name__ == "__main__":
    main()