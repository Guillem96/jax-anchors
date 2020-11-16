from typing import Iterable, Tuple

import jax
import jax.numpy as np

from anchors_jax.typing import Tensor


def sparse_cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Examples
    --------
    >>> target = jax.random.randint(key, shape=(10,), maxval=3, minval=0)
    >>> predictions = jax.nn.softmax(jax.random.uniform(key, shape=(10, 3)))
    >>> loss = aj.nn.losses.sparse_cross_entropy(target, predictions)
    """
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true = jax.nn.one_hot(y_true, y_pred.shape[-1])
    return -np.sum(y_true * np.log(y_pred), -1)


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    epsilon = 1e-8

    y_true = y_true.astype('float32')
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = y_true * np.log(y_pred + epsilon)
    loss += (1. - y_true) * np.log(1. - y_pred + epsilon)
    return -loss


def focal_loss(y_true: Tensor, y_pred: Tensor, 
               alpha: float = 0.25, gamma: float = 2.0) -> Tensor:

    alpha_factor = np.ones_like(y_true) * alpha
    alpha_factor = np.where(y_true > 0, alpha_factor, 1 - alpha_factor)
    focal_weight = np.where(y_true > 0, 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma

    return focal_weight * binary_cross_entropy(y_true, y_pred)


def smooth_l1(y_true: Tensor, y_pred: Tensor, beta: float = 1.) -> Tensor:
    y_true = y_true.astype('float32')
    y_pred = y_pred.astype('float32')

    error = y_true - y_pred
    abs_error = np.abs(error)
    return np.where(abs_error < beta,
                    .5 * abs_error ** 2 / beta,
                    abs_error - .5 * beta)


def l2_loss(params: Iterable[Tensor]) -> float:
  return 0.5 * sum(np.sum(np.square(p)) for p in params)


def ssd_loss(cls_true: Tensor,
             reg_true: Tensor,
             cls_pred: Tensor,
             reg_pred: Tensor) -> Tuple[float, float]:
    """
    Computes object detection loss for ssd model

    Parameters
    ----------
    cls_true: Tensor of shape [BATCH, N]
        Anchors labels
    reg_true: Tensor of shape [BATCH, N, 4]
        Ground truth regressors
    cls_pred: Tensor of shape [BATCH, N, NUM_CLASSES + 1]
        Predicted probabilities distributions for classification (after softmax)
    reg_pred: Tensor of shape [BATCH, N, 4]
        Predicted regressors

    Returns
    -------
    Tuple[float, float]
        Classification and regression loss
    """
    bs = cls_true.shape[0]

    # Get the positive and negative mask to pick the corresponding samples
    # This will also help to avoid the ignored boxes labeled with -1
    positive_mask = (cls_true > 0).astype('float32')
    negative_mask = (cls_true == 0).astype('float32')

    # As described in the ssd paper, we want three more times negative samples
    # than positive ones
    n_positives = np.sum(positive_mask, axis=1).astype('int32')
    n_negatives = n_positives * 3.

    # Apply sparse categorical cross entropy to all samples
    cls_loss = jax.vmap(sparse_cross_entropy)(cls_true, cls_pred)

    # Filter only the positive ones and sum the batch loss to obtain a 
    # tensor of shape [batch_size] containing the lost for each batch image
    pos_cls_loss = cls_loss * positive_mask.reshape(bs, -1)
    pos_cls_loss = np.sum(pos_cls_loss)

    # Pick the negatives' sample loss and compute the mask to keep only the
    # highest loss samples
    neg_cls_loss = cls_loss * negative_mask.reshape(bs, -1)
    neg_cls_loss = -np.sort(-neg_cls_loss)
    neg_top_mask = _ssd_compute_negative_mask(neg_cls_loss, n_negatives)
    neg_cls_loss = neg_top_mask * neg_cls_loss

    # Sum the loss over the batch (same situation as the positive samples)
    neg_cls_loss = np.sum(neg_cls_loss, axis=-1)

    # Sum positive and negative losses and normalize the sum over batch dividing
    # by the number of positives
    cls_loss = pos_cls_loss + neg_cls_loss / np.maximum(1., n_positives)

    # We only compute the smooth l1 loss for positive samples
    reg_loss = smooth_l1(reg_true, reg_pred)
    reg_loss = np.sum(reg_loss, axis=-1) # Sum the loss of the four coordinates
    reg_loss = reg_loss * positive_mask.reshape(bs, -1) # Filter out negatives
    reg_loss = np.sum(reg_loss, axis=-1) / np.maximum(1., n_positives)

    return np.sum(cls_loss), np.sum(reg_loss)


def _ssd_compute_negative_mask(negative_losses: Tensor, 
                               n_negatives: Tensor) -> Tensor:
    """
    Examples
    --------
    >>> losses = np.array([[.5, .2, .1], [.9, .7, .3]])
    >>> n_negatives = np.array([2, 1])
    >>> mask = _compute_negative_mask(losses, n_negatives)
    >>> mask
    np.array([1., 1., 0.], [1., 0., 0.])
    """
    bs = negative_losses.shape[0]
    neg_top_mask = np.zeros_like(negative_losses)
    n_negatives = n_negatives.astype('int32')

    # TODO: Wait for jax.ops.index[np.arange(bs), n_negatives]

    for i in range(bs):
        neg_top_mask = jax.ops.index_update(
            neg_top_mask, jax.ops.index[i, n_negatives[i]], 1.)

    neg_top_mask = 1 - neg_top_mask.cumsum(-1)
    return neg_top_mask.astype('float32')
