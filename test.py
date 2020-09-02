import jax
import anchors_jax as aj

faster_rcnn_init_fn, faster_rcnn_fw_fn = aj.zoo.FasterRCNN(num_classes=2, k=9)

rng = jax.random.PRNGKey(0)
out_shape, params = faster_rcnn_init_fn(rng, (-1, 720, 1080, 3))

