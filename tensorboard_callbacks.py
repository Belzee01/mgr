import os

import numpy as np
import tensorflow as tf

from tensor_cofig import n_classes

import cv2
import io
import numpy as np
import os
import shutil
import tensorflow as tf
from PIL import Image
from skimage.io import imsave

from config import model_name, imshape, labels, hues, n_classes


#
#
# class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
#     def __init__(self, logbase, **kwargs):
#         tmp = os.path.join(logbase, 'metrics')
#         if os.path.exists(tmp):
#             shutil.rmtree(tmp)
#             os.mkdir(tmp)
#         # Make the original `TensorBoard` log to a subdirectory 'training'
#         training_log_dir = os.path.join(logbase, 'metrics', model_name + '_train')
#         super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
#
#         # Log the validation metrics to a separate subdirectory
#         val_log_dir = os.path.join(logbase, 'metrics', model_name + '_val')
#         self.val_log_dir = val_log_dir
#
#     def set_model(self, model):
#         # Setup writer for validation metrics
#         self.val_writer = tf.summary.FileWriter(self.val_log_dir)
#         super(TrainValTensorBoard, self).set_model(model)
#
#     def on_epoch_end(self, epoch, logs=None):
#         # Pop the validation logs and handle them separately with
#         # `self.val_writer`. Also rename the keys so that they can
#         # be plotted on the same figure with the training metrics
#         logs = logs or {}
#         val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
#         for name, value in val_logs.items():
#             summary = tf.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value.item()
#             summary_value.tag = name
#             self.val_writer.add_summary(summary, epoch)
#         self.val_writer.flush()
#
#         # Pass the remaining logs to `TensorBoard.on_epoch_end`
#         logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
#         super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)


class TensorBoardMask2(tf.keras.callbacks.Callback):
    def __init__(self, original_images, log_dir, log_freq):
        super().__init__()
        self.log_freq = log_freq
        self.im_summaries = []
        self.global_batch = 0
        self.images = original_images
        self.logdir = log_dir + "/images"
        self.writer = tf.summary.create_file_writer(self.logdir)
        self.step = 0

    def log_mask(self):
        image_summaries = []
        for im in self.images:
            mask = self.predict(im)
            image_summaries.append(mask)

        if len(mask.shape) == 2:
            output_shape = 1
        else:
            output_shape = 3
        image_summary = np.reshape(image_summaries[:], (-1, mask.shape[0], mask.shape[1], output_shape))
        with self.writer.as_default():
            tf.summary.image('Training data', image_summary, max_outputs=self.images.shape[0], step=self.step)
            self.writer.flush()
        self.step += 1

    def add_masks(self, pred):
        blank = np.zeros(shape=imshape, dtype=np.uint8)

        for i, label in enumerate(labels):
            hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
            sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
            val = pred[:, :, i].astype(np.uint8)

            im_hsv = cv2.merge([hue, sat, val])
            im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
            blank = cv2.add(blank, im_rgb)

        return blank

    def predict(self, im):
        if imshape[2] == 1:
            im = im.reshape(im.shape[0], imshape[1], 1)
        elif imshape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.reshape(im.shape[0], im.shape[1], 3)
        im = np.expand_dims(im, axis=0)
        pred = self.model.predict(im)
        pred = np.squeeze(pred) * 255.0
        if n_classes == 1:
            mask = np.array(pred, dtype=np.uint8)
        elif n_classes > 1:
            mask = self.add_masks(pred)
        return mask

    def on_epoch_end(self, epoch, logs={}):
        if int(epoch % self.log_freq) != 0:
            return
        self.log_mask()
        self.global_batch += self.log_freq
