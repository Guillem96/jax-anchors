import sys
import click

import jax
import jax.numpy as np

from PIL import Image

import tensorflow as tf

sys.path.append('..')
import anchors_jax as aj


@click.command()
@click.argument('im_path')
def main(im_path):
    rng = jax.random.PRNGKey(0)

    init_fn, forward_fn = aj.zoo.VGG16(1000)

    im = Image.open(im_path).convert('RGB').resize((224, 224))
    im = aj.image.caffe_preprocess(im)
    im = np.expand_dims(im, 0)

    out_shape, _ = init_fn(rng, (-1, 224, 224, 3))
    params = aj.zoo.VGG16_imagenet_weights()
    prediction = forward_fn(params, im)

    print(tf.keras.applications.imagenet_utils.decode_predictions(prediction))


if __name__ == "__main__":
    main()
