import os

import numpy as np
import tensorflow as tf

from tensor_cofig import n_classes


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
