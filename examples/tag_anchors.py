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
    
    # Load boxes from labelme rectangles
    annot = json.load(open(annot_path))

    labels = [o['label'] for o in annot['shapes']]
    class_2_idx = _get_label_mapping(labels)
    labels_idx = np.array([class_2_idx[o] for o in labels])

    boxes = [sum(o['points'], []) for o in annot['shapes']]
    boxes = np.array(boxes).astype('int32')

    im = aj.viz.draw_boxes(im, 
                           boxes=boxes, 
                           colors=aj.viz.colored(labels),
                           labels=labels)

    anchors = aj.faster_rcnn.generate_anchors(im.size[::-1],
                                              stride=8,
                                              scales=(32**2, 64**2, 128**2))
    anchors = anchors.reshape(-1, 4)
    
    cls_labels, _ = aj.faster_rcnn.detect_tag_anchors(
        anchors=anchors, boxes=boxes, labels=labels_idx)
    cls_labels = cls_labels.reshape(-1)

    positive_anchors = anchors[cls_labels > 0.]
    anchors_colors = aj.viz.colored(cls_labels[cls_labels > 0.].tolist())

    aj.viz.draw_boxes(im, boxes=positive_anchors, colors=anchors_colors).show()


def _get_label_mapping(labels: List[str]) -> Mapping[str, int]:
    classes = set(labels)
    return {c: i for i, c in enumerate(classes, start=1)}


if __name__ == "__main__":
    main()