import sys
import json
import click
from typing import List, Mapping

from PIL import Image

import jax.numpy as np

sys.path.append('..')
import anchors_jax as aj


@click.command()
@click.argument('im_path')
@click.argument('annot_path')
def main(im_path: str, annot_path: str) -> None:

    im = Image.open(im_path)
    initial_size = im.size[::-1]
    im = im.resize((1000, 600))
    h, w = im.size[::-1]
    h = h // 16
    w = w // 16

        # Load labelme annoations
    boxes, labels_idx, label_2_idx = aj.io.load_labelme_annot(
        annot_path, normalize_boxes=False)
    idx_2_label = {v: k for k, v in label_2_idx.items()}
    labels = [idx_2_label[i] for i in labels_idx.tolist()]

    im = aj.viz.draw_boxes(im, 
                           boxes=boxes, 
                           colors=aj.viz.colored(labels),
                           labels=labels)

    anchors = aj.faster_rcnn.generate_anchors((h, w), stride=16)

    cls_labels, _ = aj.faster_rcnn.detect_tag_anchors(
        anchors=anchors, boxes=boxes, labels=labels_idx, im_size=initial_size)
    cls_labels = cls_labels.reshape(-1)

    positive_anchors = anchors[cls_labels > 0.]
    anchors_colors = aj.viz.colored(cls_labels[cls_labels > 0.].tolist())

    aj.viz.draw_boxes(im, boxes=positive_anchors, colors=anchors_colors).show()


if __name__ == "__main__":
    main()