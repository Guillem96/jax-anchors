import h5py
import pickle

import jax
import jax.numpy as np

import haiku as hk

import anchors_jax as aj


def _forward(image, is_training):
    net = aj.zoo.SSD(pretrained_backbone=False, 
                     num_classes=10, k=[6, 6, 6, 6, 6, 5])
    return net(image, training=is_training)

rng = jax.random.PRNGKey(0)
im_in = jax.random.uniform(rng, shape=(2, 300, 300, 3))
ssd = hk.transform_with_state(_forward)
ssd = hk.without_apply_rng(ssd)
params, state = ssd.init(rng, im_in, is_training=True)

print('Initial n params:', len(params))

converted_params = {}

f = h5py.File('VGG_coco_SSD_300x300_iter_400000.h5', 'r')
print('Pretrained n params:', len(f.keys()))

mapping = {
    'conv1_1': 'ssd/vgg16/~features/conv2_d',
    'conv1_2': 'ssd/vgg16/~features/conv2_d_1',
    'conv2_1': 'ssd/vgg16/~features/conv2_d_2',
    'conv2_2': 'ssd/vgg16/~features/conv2_d_3',
    'conv3_1': 'ssd/vgg16/~features/conv2_d_4',
    'conv3_2': 'ssd/vgg16/~features/conv2_d_5',
    'conv3_3': 'ssd/vgg16/~features/conv2_d_6',
    'conv4_1': 'ssd/vgg16/~features/conv2_d_7',
    'conv4_2': 'ssd/vgg16/~features/conv2_d_8',
    'conv4_3': 'ssd/vgg16/~features/conv2_d_9',
    'conv5_1': 'ssd/vgg16/~features/conv2_d_10',
    'conv5_2': 'ssd/vgg16/~features/conv2_d_11',
    'conv5_3': 'ssd/vgg16/~features/conv2_d_12',
    
    'fc6': 'ssd/conv2_d',
    'fc7': 'ssd/conv2_d_1',

    'conv6_1': 'ssd/~_additional_conv/additional_conv_8_1',
    'conv6_2': 'ssd/~_additional_conv/additional_conv_8_2',

    'conv7_1': 'ssd/~_additional_conv/additional_conv_9_1',
    'conv7_2': 'ssd/~_additional_conv/additional_conv_9_2',
    
    'conv8_1': 'ssd/~_additional_conv/additional_conv_10_1',
    'conv8_2': 'ssd/~_additional_conv/additional_conv_10_2',

    'conv9_1': 'ssd/~_additional_conv/additional_conv_11_1',
    'conv9_2': 'ssd/~_additional_conv/additional_conv_11_2',

    'conv4_3_norm': 'ssd/l2_norm',
    'conv4_3_norm_mbox_conf': 'ssd/~_head/head_fm_0_cls',
    'conv4_3_norm_mbox_loc': 'ssd/~_head/head_fm_0_reg',

    'fc7_mbox_conf': 'ssd/~_head/head_fm_1_cls',
    'fc7_mbox_loc': 'ssd/~_head/head_fm_1_reg',

    'conv6_2_mbox_conf': 'ssd/~_head/head_fm_2_cls',
    'conv6_2_mbox_loc': 'ssd/~_head/head_fm_2_reg',

    'conv7_2_mbox_conf': 'ssd/~_head/head_fm_3_cls',
    'conv7_2_mbox_loc': 'ssd/~_head/head_fm_3_reg',

    'conv8_2_mbox_conf': 'ssd/~_head/head_fm_4_cls',
    'conv8_2_mbox_loc': 'ssd/~_head/head_fm_4_reg',

    'conv9_2_mbox_conf': 'ssd/~_head/head_fm_5_cls',
    'conv9_2_mbox_loc': 'ssd/~_head/head_fm_5_reg',
}

for i, layer in enumerate(f.keys()):
    keras_params = f[layer][layer]
    key = mapping[layer]
    if ('norm' in layer and 
            layer != 'conv4_3_norm_mbox_conf' and 
            layer != 'conv4_3_norm_mbox_loc'):

        converted_params[key] = {
            'gamma':  np.asarray(keras_params['weights_0:0'])
        }
    else:
        converted_params[key] = {
            'w': np.asarray(keras_params['kernel:0']),
            'b': np.asarray(keras_params['bias:0'])
        }

pickle.dump(converted_params, open('ssd_coco.jax', 'wb'))