# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import click

from PIL import Image

import anchors_jax as aj


@click.command()
@click.argument('im_path')
@click.option('--save-path', type=str, default=None, 
              help="If set, the image will be stored in the specified path")
def main(im_path: str, save_path: str):
    im = Image.open(im_path)
    im = im.resize((720, 720))

    print("Generating anchors...")
    feature_maps_shapes = [(32, 32), (16, 16), (8, 8), (4, 4)]
    anchors = aj.ssd.generate_anchors(feature_maps_shapes=feature_maps_shapes)
    anchors = [aj.boxes.cxcywh_to_xyxy(o) for o in anchors]

    anchors = [aj.boxes.scale_boxes(o, im.size[::-1]) for o in anchors]

    print('Drawing boxes...')
    im = aj.viz.draw_boxes(im, anchors[0][6:12],
                           boxes_width=2, 
                           colors=[aj.viz.Color.Red, 
                                   aj.viz.Color.Green, 
                                   aj.viz.Color.Blue,
                                   aj.viz.Color.White,
                                   aj.viz.Color.Purple,
                                   aj.viz.Color.Yellow])
    im.show()
    if save_path is not None:
        im.save(save_path)


if __name__ == "__main__":
    main()