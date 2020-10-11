import itertools
from typing import List, Tuple, Union

import jax
import jax.numpy as np
import haiku as hk

import anchors_jax as aj

from anchors_jax.typing import Tensor


class SSD(hk.Module):

    def __init__(self,
                 num_classes: int,
                 k: Union[int, List[int]],
                 pretrained: bool = False,
                 pretrained_backbone: bool = True):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.k = k
        if not isinstance(k, list):
            self.k = [k]

        self.pretrained_backbone = pretrained_backbone
        self.xavier_init_fn = aj.nn.initializers.XavierUniform()

        if pretrained:
            bp, ssd_p = SSD_VGG_VOC_weights()
            self.backbone_initial_weights = bp
            self.ssd_initial_weights = ssd_p
        elif pretrained_backbone:
            self.backbone_initial_weights = aj.zoo.VGG16_imagenet_weights()
            self.ssd_initial_weights = None
        else:
            self.backbone_initial_weights = None
            self.ssd_initial_weights = None

    def _head(self, x: Tensor, k: int, name: str) -> Tuple[Tensor, Tensor]:
        override_weights = (self.ssd_initial_weights is not None and
                            self.num_classes == 21) # When COCO

        b_init_fn = aj.nn.initializers.PriorProbability(0.01)
        if override_weights:
            params_t = 'ssd/~_head/head_{}_{}'
            params = self.ssd_initial_weights
            init_fns = {
                'clf': dict(w=hk.initializers.Constant(
                                params[params_t.format(name, 'cls')]['w']),
                            b=hk.initializers.Constant(
                                params[params_t.format(name, 'cls')]['b'])),
                'reg': dict(w=hk.initializers.Constant(
                                params[params_t.format(name, 'reg')]['w']),
                            b=hk.initializers.Constant(
                                params[params_t.format(name, 'reg')]['b']))
            }
        else:
            init_fns = {
                'clf': dict(w=self.xavier_init_fn, b=b_init_fn),
                'reg': dict(w=self.xavier_init_fn, b=None)
            }

        clf = hk.Conv2D(k * self.num_classes, kernel_shape=3, 
                        padding="SAME",
                        b_init=init_fns['clf']['b'],
                        w_init=init_fns['clf']['w'], 
                        name=f"head_{name}_cls")(x)
        clf = clf.reshape(x.shape[0], -1, self.num_classes)
        clf = jax.nn.softmax(clf, axis=-1)

        reg = hk.Conv2D(k * 4, kernel_shape=3, padding="SAME",
                        w_init=init_fns['reg']['w'],
                        b_init=init_fns['reg']['b'],
                        name=f"head_{name}_reg")(x)
        reg = reg.reshape(x.shape[0], -1, 4)

        return clf, reg

    def _additional_conv(self, 
                         x: Tensor, 
                         interim_channels: int, 
                         out_channels: int,
                         training: bool,
                         stride: int, name: str) -> Tensor:

        if self.ssd_initial_weights is not None:
            params_t = 'ssd/~_additional_conv/additional_{}_{}'
            params = self.ssd_initial_weights
            init_fns = [dict(w=hk.initializers.Constant(
                                params[params_t.format(name, i)]['w']),
                             b=hk.initializers.Constant(
                                params[params_t.format(name, i)]['b']))
                        for i in range(1, 3)]
        else:
            init_fns = [dict(w=self.xavier_init_fn, b=None) 
                        for _ in range(1, 3)]

        x = hk.Conv2D(output_channels=interim_channels, 
                      kernel_shape=1,
                      stride=1, 
                      padding='SAME',
                      w_init=init_fns[0]['w'],
                      b_init=init_fns[0]['b'],
                      name=f'additional_{name}_1')(x)
        x = jax.nn.relu(x)

        x = hk.Conv2D(output_channels=out_channels, 
                      kernel_shape=3,
                      stride=stride,
                      padding='SAME' if stride == 2 else 'VALID',
                      w_init=init_fns[1]['w'],
                      b_init=init_fns[1]['b'],
                      name=f'additional_{name}_2')(x)
        x = jax.nn.relu(x)

        return x

    def __call__(self, x: Tensor, 
                 training: bool = False) -> Tuple[Tensor, Tensor]:
        params = self.ssd_initial_weights
        x = aj.zoo.VGG16(include_top=False, 
                         pretrained=False,
                         initial_weights=self.backbone_initial_weights,
                         output_feature_maps=True)(x)

        conv4_3, x = x[-2:]
        conv4_3 = aj.nn.layers.L2Norm(
            init_fn=(hk.initializers.Constant(params['ssd/l2_norm']['gamma']) 
                     if params is not None 
                     else hk.initializers.Constant(20.)))(conv4_3)

        x = hk.MaxPool(window_shape=(1, 3, 3, 1), strides=1, padding='SAME')(x)

        # Replace fully connected by FCN
        conv_6 = hk.Conv2D(
            output_channels=1024, 
            kernel_shape=3, 
            stride=1,
            rate=6,
            w_init=(hk.initializers.Constant(params['ssd/conv2_d']['w'])
                    if params is not None else self.xavier_init_fn),
            b_init=(hk.initializers.Constant(params['ssd/conv2_d']['b'])
                    if params is not None else None),
            padding='SAME')(x)
        conv_6 = jax.nn.relu(conv_6)

        conv7 = hk.Conv2D(
            output_channels=1024, 
            kernel_shape=1, 
            stride=1,
            w_init=(hk.initializers.Constant(params['ssd/conv2_d_1']['w'])
                    if params is not None else self.xavier_init_fn),
            b_init=(hk.initializers.Constant(params['ssd/conv2_d_1']['b'])
                    if params is not None else None),
            padding='SAME')(conv_6)
        conv7 = jax.nn.relu(conv7)

        # Build additional features
        conv8_2 = self._additional_conv(conv7, 256, 512, stride=2,
                                        training=training, 
                                        name='conv_8')
        conv9_2 = self._additional_conv(conv8_2, 128, 256, stride=2,
                                        training=training, 
                                        name='conv_9')
        conv10_2 = self._additional_conv(conv9_2, 128, 256, stride=1,
                                         training=training, 
                                         name='conv_10')
        conv11_2 = self._additional_conv(conv10_2, 128, 256, stride=1,
                                         training=training, 
                                         name='conv_11')

        detection_features = [conv4_3, conv7, conv8_2, 
                              conv9_2, conv10_2, conv11_2]

        detection_features = zip(itertools.cycle(self.k), detection_features)

        clf, reg = list(zip(*[self._head(o, k=k, name=f"fm_{i}") 
                              for i, (k, o) in enumerate(detection_features)]))

        return np.concatenate(clf, axis=1), np.concatenate(reg, axis=1) 


def SSD_VGG_VOC_weights():
    import pickle

    params = pickle.load(open('../ssd_voc.jax', 'rb'))
    backbone_params = {k:v for k, v in params.items() if 'vgg' in k}
    backbone_params = {k.replace('ssd/', '').replace('~features/', ''): v
                       for k,v in backbone_params.items()}

    ssd_params = {k:v for k, v in params.items() if 'vgg' not in k}

    return backbone_params, ssd_params
