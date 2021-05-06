import datetime
import os.path as osp
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from metrics import dice
from tensorboard_callbacks import TensorBoardMask2


class U_Net:
    @staticmethod
    def preprocess_input(x):
        x /= 255
        return x

    @staticmethod
    def create(n_classes=1, base=2, predefined=False):
        if n_classes == 1:
            loss = 'binary_crossentropy'
            final_act = 'sigmoid'
        elif n_classes > 1:
            loss = 'categorical_crossentropy'
            final_act = 'softmax'

        inputs = tf.keras.layers.Input(INPUT_SHAPE)
        b = base

        # Contraction path
        # Step 1
        converted_inputs = tf.keras.layers.Lambda(U_Net.preprocess_input)(inputs)
        c_1 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            converted_inputs)
        c_1 = tf.keras.layers.Dropout(0.1)(c_1)
        c_1 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            c_1)

        # Step 2
        p_1 = tf.keras.layers.MaxPooling2D((2, 2))(c_1)
        c_2 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            p_1)
        c_2 = tf.keras.layers.Dropout(0.1)(c_2)
        c_2 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(c_2)

        # Step 3
        p_2 = tf.keras.layers.MaxPooling2D((2, 2))(c_2)
        c_3 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            p_2)
        c_3 = tf.keras.layers.Dropout(0.1)(c_3)
        c_3 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(c_3)

        # Step 4
        p_3 = tf.keras.layers.MaxPooling2D((2, 2))(c_3)
        c_4 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            p_3)
        c_4 = tf.keras.layers.Dropout(0.1)(c_4)
        c_4 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(c_4)

        # Step 5
        p_4 = tf.keras.layers.MaxPooling2D((2, 2))(c_4)
        c_5 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            p_4)
        c_5 = tf.keras.layers.Dropout(0.1)(c_5)
        c_5 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(c_5)

        # Step 6
        p_5 = tf.keras.layers.MaxPooling2D((2, 2))(c_5)
        c_6 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            p_5)
        c_6 = tf.keras.layers.Dropout(0.1)(c_6)
        c_6 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            c_6)

        # Step 6
        p_6 = tf.keras.layers.MaxPooling2D((2, 2))(c_6)
        c_7 = tf.keras.layers.Conv2D(2 ** (b + 6), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            p_6)
        c_7 = tf.keras.layers.Dropout(0.1)(c_7)
        c_7 = tf.keras.layers.Conv2D(2 ** (b + 6), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            c_7)

        # Expansion path
        # Step 1
        u_6 = tf.keras.layers.Conv2DTranspose(2 ** (b + 5), (2, 2), strides=(2, 2), padding='same')(c_7)
        u_6 = tf.keras.layers.concatenate([u_6, c_6])
        c_8 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            u_6)
        c_8 = tf.keras.layers.Dropout(0.1)(c_8)
        c_8 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            c_8)

        # Step 2
        u_7 = tf.keras.layers.Conv2DTranspose(2 ** (b + 4), (2, 2), strides=(2, 2), padding='same')(c_8)
        u_7 = tf.keras.layers.concatenate([u_7, c_5])
        c_9 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            u_7)
        c_9 = tf.keras.layers.Dropout(0.1)(c_9)
        c_9 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                     padding='same')(
            c_9)

        # Step 3
        u_8 = tf.keras.layers.Conv2DTranspose(2 ** (b + 3), (2, 2), strides=(2, 2), padding='same')(c_9)
        u_8 = tf.keras.layers.concatenate([u_8, c_4])
        c_10 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                      padding='same')(
            u_8)
        c_10 = tf.keras.layers.Dropout(0.1)(c_10)
        c_10 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                      padding='same')(
            c_10)

        # Step 4
        u_9 = tf.keras.layers.Conv2DTranspose(2 ** (b + 2), (2, 2), strides=(2, 2), padding='same')(c_10)
        u_9 = tf.keras.layers.concatenate([u_9, c_3])
        c_11 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                      padding='same')(
            u_9)
        c_11 = tf.keras.layers.Dropout(0.1)(c_11)
        c_11 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                      padding='same')(
            c_11)

        # Step 5
        u_10 = tf.keras.layers.Conv2DTranspose(2 ** (b + 1), (2, 2), strides=(2, 2), padding='same')(c_11)
        u_10 = tf.keras.layers.concatenate([u_10, c_2])
        c_12 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                      padding='same')(
            u_10)
        c_12 = tf.keras.layers.Dropout(0.1)(c_12)
        c_12 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                      padding='same')(
            c_12)

        # Step 6
        u_11 = tf.keras.layers.Conv2DTranspose(2 ** b, (2, 2), strides=(2, 2), padding='same')(c_12)
        u_11 = tf.keras.layers.concatenate([u_11, c_1])
        c_13 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal',
                                      padding='same')(
            u_11)
        c_13 = tf.keras.layers.Dropout(0.1)(c_13)
        c_13 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal',
                                      padding='same')(
            c_13)

        outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation=final_act)(c_13)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss=loss,
                      metrics=[dice])
        model.summary()

        return model


class DataGenerator:
    @staticmethod
    def get_data_sets(img_width, img_height, training_length, testing_length, img_channels=3):
        train_inputs = np.zeros((training_length + testing_length, img_height, img_width, img_channels), dtype=np.uint8)
        train_labels = np.zeros((training_length + testing_length, img_height, img_width, 1), dtype=np.uint8)

        face_data = './CelebAMask-HQ/CelebA-HQ-img'
        mask_data = './CelebAMask-HQ/mask'

        # Load input data
        for i, id_ in tqdm(enumerate(range(0, training_length + testing_length)),
                           total=training_length + testing_length):
            train_filename = str(i) + '.jpg'
            mask_filename = str(i) + '.png'
            train_path = osp.join(face_data, train_filename)
            mask_path = osp.join(mask_data, mask_filename)
            input = imread(train_path)[:, :, :img_channels]
            input = resize(input, (img_height, img_width), mode='constant', preserve_range=True)
            train_inputs[i] = input
            # Set singular label
            mask = np.zeros((img_height, img_width))
            sep_mask = np.array(Image.open(mask_path).convert('P'))
            sep_mask = resize(sep_mask, (img_height, img_width), mode='constant', preserve_range=True)
            mask[sep_mask == 255] = 1
            mask = np.expand_dims(resize(mask, (img_height, img_width), mode='constant',
                                         preserve_range=True), axis=-1)
            train_labels[i] = mask
        # testing inputs, testing labels, training inputs, training labels
        return train_inputs[training_length: training_length + testing_length], \
               train_labels[training_length: training_length + testing_length], \
               train_inputs[:TRAIN_LENGTH], train_labels[:TRAIN_LENGTH]


seed = 42
np.random.seed = seed

# Input dimensions
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# Input data
TRAIN_LENGTH = 1000

face_data = './CelebAMask-HQ/CelebA-HQ-img'
mask_data = './CelebAMask-HQ/mask'

# Load test data
TEST_LENGTH = 2
test_inputs, test_labels, train_inputs, train_labels = \
    DataGenerator.get_data_sets(IMG_WIDTH, IMG_HEIGHT, TRAIN_LENGTH, TEST_LENGTH, IMG_CHANNELS)
# Model
model = U_Net.create(base=2)

# Model checkpoints
checkpoint = tf.keras.callbacks.ModelCheckpoint('model_mask.h5', verbose=1, save_best_only=True)

# Model callbacks
logdir = "logs/fit/unet" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    # checkpoint,
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
    TensorBoardMask2(original_images=test_inputs, log_dir=logdir, log_freq=5)
]

# Model learning
result = model.fit(train_inputs, train_labels, validation_split=0.1, batch_size=16, epochs=200, callbacks=callbacks)

for key in ['loss', 'val_loss']:
    plt.plot(result.history[key], label=key)
plt.legend()
plt.show()

model.save('saved_model/u-net')

idx = random.randint(0, len(train_inputs))

preds_train = model.predict(train_inputs[:int(train_inputs.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(train_inputs[int(train_inputs.shape[0] * 0.9):], verbose=1)
# preds_test = model.predict(, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
# preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
f, axarr = plt.subplots(3, 1)
ix = random.randint(0, len(preds_train_t))
axarr[0].imshow(train_inputs[:int(train_inputs.shape[0] * 0.9)][ix])
axarr[1].imshow(np.squeeze(train_labels[int(train_inputs.shape[0] * 0.9):][ix]))
axarr[2].imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
f, axarr2 = plt.subplots(3, 1)
ix = random.randint(0, len(preds_train_t))
axarr2[0].imshow(train_inputs[:int(train_inputs.shape[0] * 0.9)][ix])
axarr2[1].imshow(np.squeeze(train_labels[int(train_inputs.shape[0] * 0.9):][ix]))
axarr2[2].imshow(np.squeeze(preds_val_t[ix]))
plt.show()
