import jax
import jax.numpy as np

import haiku as hk

from PIL import Image
import anchors_jax as aj
import tensorflow as tf

def _forward(image):
    net = aj.zoo.VGG16(include_top=True, pretrained=True)
    return net(image)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    print('Loading image...')
    im = Image.open('images/dog.jpeg').resize((224, 224))
    im = aj.image.caffe_preprocess(im)

    print('hk.transform...')
    vgg_forward_fn = hk.transform(_forward)

    # print('Stuck in init')
    params = vgg_forward_fn.init(rng, np.expand_dims(im, 0))
    preds = vgg_forward_fn.apply(params, rng, np.expand_dims(im, 0))
    print(tf.keras.applications.imagenet_utils.decode_predictions(preds))
