# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import click

from PIL import Image

import anchors_jax.faster_rcnn as aj
import anchors_jax.viz as viz


@click.command()
@click.argument('im_path')
@click.option('--scales', type=str, default=None, 
              help="Comma separated scales (areas of the anchors)")
@click.option('--stride', type=int, default=10, 
              help="Distance in pixels between sets of anchors")
@click.option('--save-path', type=str, default=None, 
              help="If set, the image will be stored in the specified path")
def main(im_path: str, scales: str, stride: int, save_path: str):
    im = Image.open(im_path)
    
    kwargs = {}
    if scales is not None:
        kwargs['scales'] = [float(eval(o)) for o in scales.split(',')]

    print("Generating anchors...")
    anchors = aj.generate_anchors((im.size[1] // stride, im.size[0] // stride), 
                                  stride=stride, **kwargs)

    print('Drawing boxes...')
    im = viz.draw_boxes(im, anchors, 
                        boxes_width=2, 
                        colors=[viz.Color.Red, viz.Color.Green, viz.Color.Blue])
    im.show()
    if save_path is not None:
        im.save(save_path)


if __name__ == "__main__":
    main()