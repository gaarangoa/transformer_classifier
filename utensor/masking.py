import tensorflow as tf
import numpy as np


def create_look_ahead_mask(size):
    """
    The look-ahead mask is used to mask the future tokens in a sequence. In other words,
    the mask indicates which entries should not be used.

    This means that to predict the third word, only the first and second word will be used.
    Similarly to predict the fourth word, only the first, second and the third word will
    be used and so on.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(seq, tar):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_masks_train(seq, tar):
    # This serves as augmentation, what is doing is removing some values ar random
    seq = tf.cast(tf.math.greater(tf.random.uniform(
        seq.shape, minval=0, maxval=10,), 5), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
