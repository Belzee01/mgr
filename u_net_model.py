import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input, _preprocess_symbolic_input

from config import color_labels, id2code
from data_generator import generate_training_set, generate_labels, heatmap_to_rgb, shuffle
from image_preprocessing import gaussian_blur, mean_filter, noise
from metrics import dice, iou_coef
from tensorboard_callbacks import TensorBoardMask2


def create(n_classes=1, base=2, pretrained=False, pretrained_model_path='', learning_rate=1e-6, metrics=[dice]):
    if n_classes == 1:
        loss = 'binary_crossentropy'
        final_act = 'sigmoid'
    elif n_classes > 1:
        loss = 'categorical_crossentropy'
        final_act = 'softmax'

    if pretrained:
        model = load_model(pretrained_model_path,
                           custom_objects={'dice': dice, 'preprocess_input': preprocess_input,
                                           '_preprocess_symbolic_input': _preprocess_symbolic_input
                                           })
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss=loss,
                      metrics=metrics)
        model.summary()
        return model

    inputs = tf.keras.layers.Input(INPUT_SHAPE)
    b = base

    converted_inputs = tf.keras.layers.Lambda(lambda x: preprocess_input(x, mode='torch'))(inputs)
    conv_1 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
        converted_inputs)
    conv_1 = tf.keras.layers.Dropout(0.2)(conv_1)
    conv_1 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
        conv_1)

    pool_1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(pool_1)
    conv_2 = tf.keras.layers.Dropout(0.2)(conv_2)
    conv_2 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(conv_2)

    pool_2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(pool_2)
    conv_3 = tf.keras.layers.Dropout(0.2)(conv_3)
    conv_3 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(conv_3)

    pool_3 = tf.keras.layers.MaxPooling2D((2, 2))(conv_3)
    conv_4 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(pool_3)
    conv_4 = tf.keras.layers.Dropout(0.2)(conv_4)
    conv_4 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(conv_4)

    pool_4 = tf.keras.layers.MaxPooling2D((2, 2))(conv_4)
    conv_5 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(pool_4)
    conv_5 = tf.keras.layers.Dropout(0.2)(conv_5)
    conv_5 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(conv_5)

    pool_5 = tf.keras.layers.MaxPooling2D((2, 2))(conv_5)
    conv_6 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(pool_5)
    conv_6 = tf.keras.layers.Dropout(0.1)(conv_6)
    conv_6 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(conv_6)

    pool_6 = tf.keras.layers.MaxPooling2D((2, 2))(conv_6)
    conv_7 = tf.keras.layers.Conv2D(2 ** (b + 6), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(pool_6)
    conv_7 = tf.keras.layers.Dropout(0.1)(conv_7)
    conv_7 = tf.keras.layers.Conv2D(2 ** (b + 6), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(conv_7)

    de_conv_1 = tf.keras.layers.Conv2DTranspose(2 ** (b + 5), (2, 2), strides=(2, 2), padding='same')(conv_7)
    de_conv_1 = tf.keras.layers.concatenate([de_conv_1, conv_6])
    conv_8 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(de_conv_1)

    conv_8 = tf.keras.layers.Dropout(0.1)(conv_8)
    conv_8 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(conv_8)

    de_conv_2 = tf.keras.layers.Conv2DTranspose(2 ** (b + 4), (2, 2), strides=(2, 2), padding='same')(conv_8)
    de_conv_2 = tf.keras.layers.concatenate([de_conv_2, conv_5])
    conv_9 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(de_conv_2)

    conv_9 = tf.keras.layers.Dropout(0.1)(conv_9)
    conv_9 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(conv_9)

    de_conv_3 = tf.keras.layers.Conv2DTranspose(2 ** (b + 3), (2, 2), strides=(2, 2), padding='same')(conv_9)
    de_conv_3 = tf.keras.layers.concatenate([de_conv_3, conv_4])
    conv_10 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(de_conv_3)

    conv_10 = tf.keras.layers.Dropout(0.1)(conv_10)
    conv_10 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(conv_10)

    de_conv_4 = tf.keras.layers.Conv2DTranspose(2 ** (b + 2), (2, 2), strides=(2, 2), padding='same')(conv_10)
    de_conv_4 = tf.keras.layers.concatenate([de_conv_4, conv_3])
    conv_11 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(de_conv_4)

    conv_11 = tf.keras.layers.Dropout(0.1)(conv_11)
    conv_11 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(conv_11)

    de_conv_5 = tf.keras.layers.Conv2DTranspose(2 ** (b + 1), (2, 2), strides=(2, 2), padding='same')(conv_11)
    de_conv_5 = tf.keras.layers.concatenate([de_conv_5, conv_2])
    conv_12 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(de_conv_5)

    conv_12 = tf.keras.layers.Dropout(0.1)(conv_12)
    conv_12 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(conv_12)

    de_conv_6 = tf.keras.layers.Conv2DTranspose(2 ** b, (2, 2), strides=(2, 2), padding='same')(conv_12)
    de_conv_6 = tf.keras.layers.concatenate([de_conv_6, conv_1])
    conv_13 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(de_conv_6)

    conv_13 = tf.keras.layers.Dropout(0.1)(conv_13)
    conv_13 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(conv_13)

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation=final_act)(conv_13)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=loss,
                  metrics=metrics)
    model.summary()

    return model


seed = 42
np.random.seed = seed

# Input dimensions
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# Input data
TRAIN_LENGTH = 9000
TEST_LENGTH = 2

train_inputs = generate_training_set(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
train_labels = generate_labels(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH)

# Image manipulations
train_inputs[:999] = [noise(noise_type="gauss", image=image) for image in train_inputs[:999]]
train_inputs[1000:1999] = [noise(noise_type="s&p", image=image) for image in train_inputs[1000:1999]]
train_inputs[2000:2999] = [noise(noise_type="poisson", image=image) for image in train_inputs[2000:2999]]
train_inputs[3000:3999] = [noise(noise_type="speckle", image=image) for image in train_inputs[3000:3999]]
#
# train_inputs[4000:4999] = [mean_filter(image) for image in train_inputs[4000:4999]]
# train_inputs[5000:5999] = [gaussian_blur(image) for image in train_inputs[5000:5999]]

# Shuffle
train_inputs, train_labels = shuffle(train_inputs, train_labels)

test_inputs = train_inputs[5:10]
test_labels = train_labels[5:10]

# Model
model = create(base=2, n_classes=len(color_labels), pretrained=False,
               pretrained_model_path='models/unet_20210515-134641.model',
               learning_rate=1e-5, metrics=[
        dice,
        'accuracy',
        iou_coef
    ])

# Model checkpoints
model_name = 'unet_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join('models', model_name + '.model'), verbose=1,
                                                monitor='dice',
                                                save_best_only=True, mode='max',
                                                save_weights_only=False, period=10)

# Model callbacks
logdir = "logs/fit/" + model_name
callbacks = [
    checkpoint,
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
    TensorBoardMask2(original_images=test_inputs, log_dir=logdir, log_freq=25)
]

# Model learning
result = model.fit(train_inputs, train_labels, validation_split=0.2, batch_size=18, epochs=600, callbacks=callbacks)

model.save('saved_models/' + model_name + '.model')

y_pred = model.predict(test_inputs)
y_predi = y_pred

for i in range(TEST_LENGTH):
    f, axarr = plt.subplots(2, 5)

    axarr[1][4].imshow(test_inputs[i])
    axarr[1][4].set_title("original")

    axarr[1][3].imshow(heatmap_to_rgb(test_labels[i], id2code))
    axarr[1][3].set_title("truth")

    axarr[1][2].imshow(heatmap_to_rgb(y_pred[i], id2code))
    axarr[1][2].set_title("prediction mask")

    for cl in range(len(color_labels)):
        if cl > 5:
            axarr[1][cl - 5].imshow(y_pred[i, :, :, cl])
            axarr[1][cl - 5].set_title("layer " + id2code[cl + 1])
        else:
            axarr[0][cl].imshow(y_pred[i, :, :, cl])
            axarr[0][cl].set_title("layer " + id2code[cl + 1])

    plt.show()
