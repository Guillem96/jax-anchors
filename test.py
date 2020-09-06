import jax
import jax.numpy as np
from jax.experimental import optix

from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

import anchors_jax as aj

def swap_xy(boxes):
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    return tf.concat([x_min, y_min, x_max, y_max], axis=1)
 
def scale_boxes(boxes, im_shape):
    h, w = im_shape
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)

    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    x_min = x_min * w
    y_min = y_min * h 
    x_max = x_max * w 
    y_max = y_max * h
    return tf.concat([x_min, y_min, x_max, y_max], axis=1)

def resize_im(im):
    im_shape = tf.shape(im)
    w = tf.cast(im_shape[1], tf.float32)
    h = tf.cast(im_shape[0], tf.float32)

    ar = w / h
    if h > w:
        size = [tf.cast(600 / ar, tf.int32), 
                tf.constant(600, dtype=tf.int32)]
    else:
        size = [tf.constant(600, dtype=tf.int32),
                tf.cast(600 * ar, tf.int32)]

    return tf.image.resize(im, size=size)

def format_boxes(boxes, im_shape):
    boxes = swap_xy(boxes)
    boxes = scale_boxes(boxes, im_shape)
    return boxes

@tf.function
def get_example(example):
    im = resize_im(example['image'])
    im_shape = tf.shape(im)
    boxes = format_boxes(example['objects']['bbox'], (im_shape[0], im_shape[1]))
    return im, boxes

# @jax.jit
def generate_anchors(im, boxes):
    h, w = im.shape[:2]
    h = h // 16
    w = w // 16

    print(h, w)
    anchors = aj.faster_rcnn.generate_anchors((h, w), stride=16)
    labels, regressors = aj.faster_rcnn.rpn_tag_anchors(anchors, boxes)

    return anchors, labels, regressors


train_ds = tfds.load('voc', split='train')
valid_ds = tfds.load('voc', split='validation')
train_ds = tfds.as_numpy(train_ds.shuffle(1024).map(get_example))
valid_ds = tfds.as_numpy(valid_ds.map(get_example))

im, boxes = next(iter(train_ds))
anchors, labels, regressors = generate_anchors(im, boxes)
positive_anchors = anchors[labels.reshape(-1) == 1.]
im = aj.viz.draw_boxes(im, boxes, boxes_width=2)
im = aj.viz.draw_boxes(im, positive_anchors, 
                       boxes_width=2, 
                       colors=['#ef9a9a', '#f48fb1', '#ce93d8',
                               '#b39ddb', '#81d4fa', '#81d4fa',
                               '#c5e1a5', '#ffe082', '#dd2c00'])
im.show()
