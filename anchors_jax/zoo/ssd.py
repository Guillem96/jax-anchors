from typing import List, Tuple

import jax
import haiku as hk

import anchors_jax as aj

from anchors_jax.typing import Tensor


class SSD(hk.Module):

    def __init__(self,
                 num_classes: int, 
                 k: int,
                 pretrained_backbone: bool = True):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.k = k
        self.pretrained_backbone = pretrained_backbone

    def _head(self, x: Tensor, name: str) -> Tuple[Tensor, Tensor]:
        clf = hk.Conv2D(self.k * self.num_classes, 
                        kernel_shape=3, padding="SAME",
                        name=f"head_{name}_cls")(x)
        clf = clf.reshape(x.shape[0], -1, self.num_classes)
        clf = jax.nn.softmax(clf, axis=-1)

        reg = hk.Conv2D(self.k * 4, kernel_shape=3, padding="SAME",
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
                      name=f'additional_{name}_1')(x)
        # TODO: BatchNorm?
        x = jax.nn.relu(x)

        x = hk.Conv2D(output_channels=out_channels, 
                      kernel_shape=3,
                      stride=stride,
                      padding='SAME' if stride == 2 else 'VALID',
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
                      padding='SAME')(x)

        conv7 = hk.Conv2D(output_channels=1, 
                          kernel_shape=1, 
                          stride=1, 
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

        return [self._head(o, name=f"fm_{'_'.join(map(str,o.shape))}") 
                for o in detection_features]
