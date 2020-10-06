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
                 pretrained_backbone: bool = True):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.k = k
        if not isinstance(k, list):
            self.k = [k]

        self.pretrained_backbone = pretrained_backbone
        self.xavier_init_fn = aj.nn.initializers.XavierUniform()

    def _head(self, x: Tensor, k: int, name: str) -> Tuple[Tensor, Tensor]:
        clf = hk.Conv2D(k * self.num_classes, 
                        kernel_shape=3, padding="SAME", with_bias=False,
                        w_init=self.xavier_init_fn,
                        name=f"head_{name}_cls")(x)
        clf = clf.reshape(x.shape[0], -1, self.num_classes)
        clf = jax.nn.softmax(clf, axis=-1)

        reg = hk.Conv2D(k * 4, kernel_shape=3, padding="SAME", with_bias=False,
                        w_init=self.xavier_init_fn,
                        name=f"head_{name}_reg")(x)
        reg = reg.reshape(x.shape[0], -1, 4)

        return clf, reg

    def _additional_conv(self, 
                         x: Tensor, 
                         interim_channels: int, 
                         out_channels: int, 
                         stride: int, name: str) -> Tensor:

        x = hk.Conv2D(output_channels=interim_channels, 
                      kernel_shape=1,
                      stride=1, 
                      padding='SAME',
                      with_bias=False,
                      w_init=self.xavier_init_fn,
                      name=f'additional_{name}_1')(x)
        # TODO: BatchNorm?
        x = jax.nn.relu(x)

        x = hk.Conv2D(output_channels=out_channels, 
                      kernel_shape=3,
                      stride=stride,
                      padding='SAME' if stride == 2 else 'VALID',
                      with_bias=False,
                      w_init=self.xavier_init_fn,
                      name=f'additional_{name}_2')(x)
        # TODO: BatchNorm?
        x = jax.nn.relu(x)

        return x

    def __call__(self, x: Tensor) -> List[Tuple[Tensor, Tensor]]:
        x = aj.zoo.VGG16(include_top=False, 
                         pretrained=self.pretrained_backbone, 
                         output_feature_maps=True)(x)

        conv4_3, x = x[-2:]

        # Replace fully connected by FCN
        conv_6 = hk.Conv2D(output_channels=1024, 
                      kernel_shape=3, 
                      stride=1,
                      w_init=self.xavier_init_fn,
                      padding='SAME')(x)

        conv7 = hk.Conv2D(output_channels=1, 
                          kernel_shape=1, 
                          stride=1,
                          w_init=self.xavier_init_fn,
                          padding='SAME')(conv_6)

        # Build additional features
        conv8_2 = self._additional_conv(conv7, 256, 512, stride=2, 
                                        name='conv_8')
        conv9_2 = self._additional_conv(conv8_2, 128, 256, stride=2, 
                                        name='conv_9')
        conv10_2 = self._additional_conv(conv9_2, 128, 256, stride=1,
                                         name='conv_10')
        conv11_2 = self._additional_conv(conv10_2, 128, 256, stride=1,
                                         name='conv_11')

        detection_features = [conv4_3, conv7, conv8_2, 
                              conv9_2, conv10_2, conv11_2]
        detection_features = zip(itertools.cycle(self.k), detection_features)

        clf, reg = zip(*[self._head(o, k=k, name=f"fm_{i}") 
                         for i, (k, o) in enumerate(detection_features)])

        return np.concatenate(clf, axis=1), np.concatenate(reg, axis=1) 
