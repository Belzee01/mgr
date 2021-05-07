import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Add, Dropout

from config import color_labels, id2code
from data_generator import generate_labels, generate_training_set, onehot_to_rgb
from metrics import dice
from tensorboard_callbacks import TensorBoardMask2


class FCN_8:
    @staticmethod
    def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    @staticmethod
    def create(input_shape: Tuple[int, int, int], n_classes=1, base=4):
        if n_classes == 1:
            loss = 'binary_crossentropy'
            final_act = 'sigmoid'
        elif n_classes > 1:
            loss = 'categorical_crossentropy'
            final_act = 'softmax'

        i = Input(shape=input_shape)
        s = tf.keras.layers.Lambda(FCN_8.preprocess_input)(i)

        ## Block 1
        x = Conv2D(2 ** base, (3, 3), activation='relu', padding='same', name='block1_conv1')(s)
        x = Dropout(0.1)(x)
        x = Conv2D(2 ** base, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        f1 = x

        # Block 2
        x = Conv2D(2 ** (base + 1), (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Dropout(0.1)(x)
        x = Conv2D(2 ** (base + 1), (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        f2 = x

        # Block 3
        x = Conv2D(2 ** (base + 2), (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(2 ** (base + 2), (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(2 ** (base + 2), (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        pool3 = x


        # Block 4
        x = Conv2D(2 ** (base + 3), (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
        x = Dropout(0.2)(x)
        x = Conv2D(2 ** (base + 3), (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(2 ** (base + 3), (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(2 ** (base + 3), (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
        x = Dropout(0.2)(x)
        x = Conv2D(2 ** (base + 3), (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(2 ** (base + 3), (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        conv6 = Conv2D(4096, (7, 7), activation='relu', padding='same', name="conv6")(pool5)
        conv6 = Dropout(0.5)(conv6)

        conv7 = Conv2D(4096, (1, 1), activation='relu', padding='same', name="conv7")(conv6)
        conv7 = Dropout(0.5)(conv7)

        pool4_n = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(pool4)
        u2 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
        u2_skip = Add()([pool4_n, u2])

        pool3_n = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(pool3)
        u4 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(u2_skip)
        u4_skip = Add()([pool3_n, u4])

        o = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), padding='same',
                            activation=final_act)(u4_skip)

        model = Model(inputs=i, outputs=o, name='fcn8')
        model.compile(optimizer=tf.keras.optimizers.Adam(8e-5),
                      loss=loss,
                      metrics=[dice, 'accuracy'])
        model.summary()

        return model


seed = 42
np.random.seed = seed

# Input dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
TEST_LENGTH = 2

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# Input data
TRAIN_LENGTH = 500

train_inputs = generate_training_set(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
train_labels = generate_labels(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH)

test_inputs = train_inputs[5:10]
test_labels = train_labels[5:10]

# Model
model = FCN_8.create(input_shape=INPUT_SHAPE, base=6, n_classes=len(color_labels))

# Model checkpoints
checkpoint = tf.keras.callbacks.ModelCheckpoint('fcn8_mask.h5', verbose=1, save_best_only=True)

# Model callbacks
logdir = "logs/fit/fcn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    # checkpoint,
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
    TensorBoardMask2(original_images=test_inputs, log_dir=logdir, log_freq=5)
]

# Model learning
result = model.fit(train_inputs, train_labels, validation_split=0.3, batch_size=4, epochs=400, callbacks=callbacks)

model.save('saved_model/fcn8')

y_pred = model.predict(test_inputs)
y_predi = y_pred

shape = (224, 224)
for i in range(TEST_LENGTH):
    img_is = (test_inputs[i])
    seg = y_predi[i]

    fig = plt.figure(figsize=(1, 3))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_is)
    ax.set_title("original")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(onehot_to_rgb(y_predi[i], id2code))
    ax.set_title("predicted class")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(onehot_to_rgb(test_labels[i], id2code))
    ax.set_title("true class")
    plt.show()
