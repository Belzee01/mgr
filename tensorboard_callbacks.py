import cv2
import numpy as np
import tensorflow as tf

from config import imshape, id2code
from data_generator import onehot_to_rgb, prediction_to_rgb


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
            tf.summary.image('Training data', image_summary, max_outputs=len(self.images), step=self.step)
            self.writer.flush()
        self.step += 1

    def add_masks(self, pred):
        return prediction_to_rgb(pred, id2code)

    def predict(self, im):
        if imshape[2] == 1:
            im = im.reshape(im.shape[0], imshape[1], 1)
        elif imshape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.reshape(im.shape[0], im.shape[1], 3)
        im = np.expand_dims(im, axis=0)
        pred = self.model.predict(im)
        mask = self.add_masks(np.squeeze(pred))
        return mask

    def on_epoch_end(self, epoch, logs={}):
        if int(epoch % self.log_freq) != 0:
            return
        self.log_mask()
        self.global_batch += self.log_freq
