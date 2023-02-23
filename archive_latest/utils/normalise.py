import numpy as np
import tensorflow as tf


def reshape_for_broadcasting(source, target):
    """Reshapes a tensor (source) to have the correct shape and dtype of the target
    before broadcasting it with MPI.
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim - 1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        # some local information
        self.local_sum = tf.zeros(self.size, tf.float32)
        self.local_sumsq = tf.zeros(self.size, tf.float32)
        self.local_count = tf.zeros(1, tf.float32)
        # get the total sum sumsq and sum count
        self.total_sum = tf.zeros(self.size, tf.float32)
        self.total_sumsq = tf.zeros(self.size, tf.float32)
        self.total_count = tf.zeros(1, tf.float32)
        # get the mean and std
        self.mean = tf.zeros(self.size, tf.float32)
        self.std = tf.ones(self.size, tf.float32)

    def update(self, x):
        x = tf.reshape(x, shape=(-1, self.size))

        self.local_sum += tf.reduce_sum(x, axis=0)
        self.local_sumsq += tf.reduce_sum(tf.math.square(x), axis=0)
        self.local_count += tf.cast(tf.shape(x)[0], self.local_count.dtype)

    def normalize(self, x, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = reshape_for_broadcasting(self.mean, x)
        std = reshape_for_broadcasting(self.std, x)
        return tf.clip_by_value((x - mean) / std, -clip_range, clip_range)

    def denormalize(self, x):
        mean = reshape_for_broadcasting(self.mean, x)
        std = reshape_for_broadcasting(self.std, x)
        return mean + x * std

    def recompute_stats(self):
        local_count = self.local_count
        local_sum = self.local_sum
        local_sumsq = self.local_sumsq

        self.total_sum += local_sum
        self.total_sumsq += local_sumsq
        self.total_count += local_count

        self.mean = self.total_sum / self.total_count
        self.std = tf.math.sqrt(tf.maximum(tf.math.square(self.eps),
                                           (self.total_sumsq / self.total_count) -
                                           tf.math.square(self.total_sum / self.total_count)))

        # reset
        self.local_sum = tf.zeros(self.size, tf.float32)
        self.local_sumsq = tf.zeros(self.size, tf.float32)
        self.local_count = tf.zeros(1, tf.float32)
