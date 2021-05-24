from tensorflow.keras import backend as keras
import tensorflow as tf


def dice(y_true, y_pred, smooth=1e-4):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)


def mean_iou(y_true, y_pred):
    y_pred = tf.round(tf.cast(y_pred, tf.int32))
    intersect = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32), axis=[1])
    union = tf.reduce_sum(tf.cast(y_true, tf.float32), axis=[1]) + tf.reduce_sum(tf.cast(y_pred, tf.float32),
                                                                                 axis=[1])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
    score = (intersection + 1.) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) +
                                   tf.reduce_sum(tf.cast(y_pred, tf.float32)) - intersection + 1.)
    return 1 - score
