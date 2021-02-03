from PIL import Image
import io
import tensorflow as tf
import os
import cv2
import numpy as np
from skimage.io import imsave
from tensor_cofig import model_name, logbase, imshape, labels, hues, n_classes
import shutil


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        tmp = os.path.join(logbase, 'metrics')
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
            os.mkdir(tmp)
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(logbase, 'metrics', model_name + '_train')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        val_log_dir = os.path.join(logbase, 'metrics', model_name + '_val')
        self.val_log_dir = val_log_dir

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)


class TensorBoardMask(tf.keras.callbacks.Callback):
    def __init__(self, original_images, log_dir, log_freq):
        super().__init__()
        self.log_freq = log_freq
        self.im_summaries = []
        self.global_batch = 0
        self.images = original_images
        self.logdir = log_dir + "/images"
        self.writer = tf.summary.create_file_writer(self.logdir)
        self.step = 0

    def _file_generator(self, path):
        files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        for fn in files:
            yield fn

    def log_mask(self):
        masks = self.predict(self.images)
        image_summaries = []
        for i, mask in enumerate(masks):
            mask = (mask[:, :, :] * 255)
            image_summaries.append(mask)
        image_summary = np.reshape(image_summaries[:],
                                   (-1, masks.shape[1], masks.shape[2], masks.shape[3]))
        with self.writer.as_default():
            tf.summary.image('Training data', image_summary, max_outputs=self.images.shape[0], step=self.step)
            self.writer.flush()
        self.step += 1

    def predict(self, images):
        pred = self.model.predict(images)
        pred = (pred > 0.5).astype(np.uint8)
        if n_classes == 1:
            mask = pred
        return mask

    def on_epoch_end(self, epoch, logs={}):
        if int(epoch % self.log_freq) != 0:
            return
        self.log_mask()
        self.global_batch += self.log_freq
