import sys
import click

import jax
import jax.numpy as np

from PIL import Image

import haiku as hk
import tensorflow as tf

sys.path.append('..')
import anchors_jax as aj


def _forward(image):
    net = aj.zoo.VGG16(include_top=True, pretrained=True)
    return net(image)


@click.command()
@click.argument('im_path')
def main(im_path):
    rng = jax.random.PRNGKey(0)

    vgg = hk.transform(_forward)
    vgg = hk.without_apply_rng(vgg)

    im = Image.open(im_path).convert('RGB').resize((224, 224))
    im = aj.image.caffe_preprocess(im)
    im = np.expand_dims(im, 0)

    params = vgg.init(rng, im)
    prediction = vgg.apply(params, im)

    print(tf.keras.applications.imagenet_utils.decode_predictions(prediction))


if __name__ == "__main__":
    main()
