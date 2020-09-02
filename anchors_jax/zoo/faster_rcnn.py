import jax
import anchors_jax as aj


def FasterRCNN(num_classes: int, 
               k: int, 
               backbone_pretrained: bool = True):

    backbone_init_fn, backbone_fw_fn = aj.zoo.VGG16(include_top=False,
                                                    pooling=None)

    rpn_init_fn, rpn_fw_fn = aj.nn.layers.FasterRCNNRPN(
        features=256, 
        num_classes=num_classes, 
        k=k)

    def init_fun(rng, input_shape):
        backbone_rng, rpn_rng = jax.random.split(rng, 2) 
        backbone_out_shape, _ = backbone_init_fn(backbone_rng, input_shape)
        backbone_params = aj.zoo.VGG16_imagenet_weights(False)

        rpn_out_shape, rpn_params = rpn_init_fn(rpn_rng, backbone_out_shape)

        return rpn_out_shape, (backbone_params, rpn_params)
    
    def apply_fun(params, x, **kwargs):
        backbone_params, rpn_params = params
        x = backbone_fw_fn(backbone_params, x, **kwargs)
        return rpn_fw_fn(rpn_params, x, **kwargs)

    return init_fun, apply_fun
