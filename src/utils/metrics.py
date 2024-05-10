import sys
import time
import math
import numpy as np
import tensorflow as tf

# class MaskedCrossEntropyCriterion(tf.keras.losses.Loss):
#     def __init__(self, ignore_index=[-100], reduce=None):
#         super(MaskedCrossEntropyCriterion, self).__init__()
#         self.padding_idx = ignore_index
#         self.reduce = reduce

#     def call(self, outputs, targets):
#         lprobs = tf.nn.log_softmax(outputs, axis=-1)
#         lprobs = tf.reshape(lprobs, [-1, lprobs.shape[-1]])

#         for idx in self.padding_idx:
#             # remove padding idx from targets to allow gathering without error (padded entries will be suppressed later)
#             # targets[targets == idx] = 0
#             targets = tf.where(targets == idx, 0, targets)

#         # pytorch: nll_loss = -lprobs.gather(dim=-1, index=targets.unsqueeze(1))
#         indices = tf.cast(tf.expand_dims(targets, 1), tf.int32)
#         print(indices)
#         nll_loss = -tf.gather(lprobs, indices=indices, axis=-1)
#         if self.reduce:
#             nll_loss = tf.reduce_sum(nll_loss)

#         return tf.squeeze(nll_loss)

def MaskedCrossEntropyCriterion(outputs, targets, ignore_index=[-100], reduce=None):
    padding_idx = ignore_index
    lprobs = tf.nn.log_softmax(outputs, axis=-1)
    lprobs = tf.reshape(lprobs, [-1, lprobs.shape[-1]])

    for idx in padding_idx:
        # remove padding idx from targets to allow gathering without error (padded entries will be suppressed later)
        # targets[targets == idx] = 0
        targets = tf.where(targets == idx, 0, targets)

    # pytorch: nll_loss = -lprobs.gather(dim=-1, index=targets.unsqueeze(1))
    indices = tf.cast(tf.expand_dims(targets, 1), tf.int32)
    indices = tf.concat([tf.expand_dims(tf.range(0, lprobs.shape[0]), -1), indices], axis=-1)
    print("probs shape", lprobs.shape)
    print("MCEC indices", indices)
    nll_loss = -tf.gather_nd(lprobs, indices=indices)
    print("nll_loss", nll_loss)
    if reduce:
        nll_loss = tf.reduce_sum(nll_loss)

    return tf.squeeze(nll_loss)


def softIoU(out, target, e=1e-6, sum_axis=1):
    num = tf.reduce_sum(out*target, axis=sum_axis, keepdims=True)
    den = tf.reduce_sum(out+target-out*target, axis=sum_axis, keepdims=True) + e
    iou = num / den

    return iou


def update_error_types(error_types, y_pred, y_true):
    # error_types['tp_i'] += (y_pred * y_true).sum(0).cpu().data.numpy()
    # error_types['fp_i'] += (y_pred * (1-y_true)).sum(0).cpu().data.numpy()
    # error_types['fn_i'] += ((1-y_pred) * y_true).sum(0).cpu().data.numpy()
    # error_types['tn_i'] += ((1-y_pred) * (1-y_true)).sum(0).cpu().data.numpy()
    # error_types['tp_all'] += (y_pred * y_true).sum().item()
    # error_types['fp_all'] += (y_pred * (1-y_true)).sum().item()
    # error_types['fn_all'] += ((1-y_pred) * y_true).sum().item()
    tp_i = tf.reduce_sum(y_pred * y_true, axis=0)
    fp_i = tf.reduce_sum(y_pred * (1-y_true), axis=0)
    fn_i = tf.reduce_sum((1-y_pred) * y_true, axis=0)
    tn_i = tf.reduce_sum((1-y_pred) * (1-y_true), axis=0)
    tp_all = tf.reduce_sum(y_pred * y_true)
    fp_all = tf.reduce_sum(y_pred * (1-y_true))
    fn_all = tf.reduce_sum((1-y_pred) * y_true)
    with tf.device("/cpu:0"):
        error_types['tp_i'] += tf.identity(tp_i).numpy()
        error_types['fp_i'] += tf.identity(fp_i).numpy()
        error_types['fn_i'] += tf.identity(fn_i).numpy()
        error_types['tn_i'] += tf.identity(tn_i).numpy()
        error_types['tp_all'] += tf.identity(tp_all).numpy()
        error_types['fp_all'] += tf.identity(fp_all).numpy()
        error_types['fn_all'] += tf.identity(fn_all).numpy()


def compute_metrics(ret_metrics, error_types, metric_names, eps=1e-10, weights=None):
    if 'accuracy' in metric_names:
        ret_metrics['accuracy'].append(np.mean((error_types['tp_i'] + error_types['tn_i']) / (error_types['tp_i'] + error_types['fp_i'] + error_types['fn_i'] + error_types['tn_i'])))
    if 'jaccard' in metric_names:
        ret_metrics['jaccard'].append(error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all'] + eps))
    if 'dice' in metric_names:
        ret_metrics['dice'].append(2*error_types['tp_all'] / (2*(error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all']) + eps))
    if 'f1' in metric_names:
        pre = error_types['tp_i'] / (error_types['tp_i'] + error_types['fp_i'] + eps)
        rec = error_types['tp_i'] / (error_types['tp_i'] + error_types['fn_i'] + eps)
        f1_perclass = 2*(pre * rec) / (pre + rec + eps)
        if 'f1_ingredients' not in ret_metrics.keys():
            ret_metrics['f1_ingredients'] = [np.average(f1_perclass, weights=weights)]
        else:
            ret_metrics['f1_ingredients'].append(np.average(f1_perclass, weights=weights))

        pre = error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + eps)
        rec = error_types['tp_all'] / (error_types['tp_all'] + error_types['fn_all'] + eps)
        f1 = 2*(pre * rec) / (pre + rec + eps)
        ret_metrics['f1'].append(f1)