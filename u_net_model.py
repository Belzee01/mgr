import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input, _preprocess_symbolic_input

from config import color_labels, id2code
from data_generator import generate_training_set, generate_labels, onehot_to_rgb, shuffle
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

    # Contraction path
    # Step 1
    converted_inputs = tf.keras.layers.Lambda(lambda x: preprocess_input(x, mode='torch'))(inputs)
    c_1 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
        converted_inputs)
    c_1 = tf.keras.layers.Dropout(0.2)(c_1)
    c_1 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
        c_1)

    # Step 2
    p_1 = tf.keras.layers.MaxPooling2D((2, 2))(c_1)
    c_2 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(p_1)
    c_2 = tf.keras.layers.Dropout(0.2)(c_2)
    c_2 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(c_2)

    # Step 3
    p_2 = tf.keras.layers.MaxPooling2D((2, 2))(c_2)
    c_3 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(p_2)
    c_3 = tf.keras.layers.Dropout(0.2)(c_3)
    c_3 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(c_3)

    # Step 4
    p_3 = tf.keras.layers.MaxPooling2D((2, 2))(c_3)
    c_4 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(p_3)
    c_4 = tf.keras.layers.Dropout(0.2)(c_4)
    c_4 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(c_4)

    # Step 5
    p_4 = tf.keras.layers.MaxPooling2D((2, 2))(c_4)
    c_5 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(p_4)
    c_5 = tf.keras.layers.Dropout(0.2)(c_5)
    c_5 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(c_5)

    # Step 6
    p_5 = tf.keras.layers.MaxPooling2D((2, 2))(c_5)
    c_6 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(p_5)
    c_6 = tf.keras.layers.Dropout(0.1)(c_6)
    c_6 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(c_6)

    # Step 6
    p_6 = tf.keras.layers.MaxPooling2D((2, 2))(c_6)
    c_7 = tf.keras.layers.Conv2D(2 ** (b + 6), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(p_6)
    c_7 = tf.keras.layers.Dropout(0.1)(c_7)
    c_7 = tf.keras.layers.Conv2D(2 ** (b + 6), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(c_7)

    # Expansion path
    # Step 1
    u_6 = tf.keras.layers.Conv2DTranspose(2 ** (b + 5), (2, 2), strides=(2, 2), padding='same')(c_7)
    u_6 = tf.keras.layers.concatenate([u_6, c_6])
    c_8 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(u_6)

    c_8 = tf.keras.layers.Dropout(0.1)(c_8)
    c_8 = tf.keras.layers.Conv2D(2 ** (b + 5), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(c_8)

    # Step 2
    u_7 = tf.keras.layers.Conv2DTranspose(2 ** (b + 4), (2, 2), strides=(2, 2), padding='same')(c_8)
    u_7 = tf.keras.layers.concatenate([u_7, c_5])
    c_9 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(u_7)

    c_9 = tf.keras.layers.Dropout(0.1)(c_9)
    c_9 = tf.keras.layers.Conv2D(2 ** (b + 4), (3, 3), activation='relu', kernel_initializer='he_normal',
                                 padding='same')(c_9)

    # Step 3
    u_8 = tf.keras.layers.Conv2DTranspose(2 ** (b + 3), (2, 2), strides=(2, 2), padding='same')(c_9)
    u_8 = tf.keras.layers.concatenate([u_8, c_4])
    c_10 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(u_8)

    c_10 = tf.keras.layers.Dropout(0.1)(c_10)
    c_10 = tf.keras.layers.Conv2D(2 ** (b + 3), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(c_10)

    # Step 4
    u_9 = tf.keras.layers.Conv2DTranspose(2 ** (b + 2), (2, 2), strides=(2, 2), padding='same')(c_10)
    u_9 = tf.keras.layers.concatenate([u_9, c_3])
    c_11 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(u_9)

    c_11 = tf.keras.layers.Dropout(0.1)(c_11)
    c_11 = tf.keras.layers.Conv2D(2 ** (b + 2), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(c_11)

    # Step 5
    u_10 = tf.keras.layers.Conv2DTranspose(2 ** (b + 1), (2, 2), strides=(2, 2), padding='same')(c_11)
    u_10 = tf.keras.layers.concatenate([u_10, c_2])
    c_12 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(u_10)

    c_12 = tf.keras.layers.Dropout(0.1)(c_12)
    c_12 = tf.keras.layers.Conv2D(2 ** (b + 1), (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(c_12)

    # Step 6
    u_11 = tf.keras.layers.Conv2DTranspose(2 ** b, (2, 2), strides=(2, 2), padding='same')(c_12)
    u_11 = tf.keras.layers.concatenate([u_11, c_1])
    c_13 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(u_11)

    c_13 = tf.keras.layers.Dropout(0.1)(c_13)
    c_13 = tf.keras.layers.Conv2D(2 ** b, (3, 3), activation='relu', kernel_initializer='he_normal',
                                  padding='same')(c_13)

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation=final_act)(c_13)

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
TRAIN_LENGTH = 12000
TEST_LENGTH = 2

train_inputs = generate_training_set(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
train_labels = generate_labels(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH)

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
result = model.fit(train_inputs, train_labels, validation_split=0.2, batch_size=18, epochs=500, callbacks=callbacks)

model.save('saved_models/' + model_name + '.model')

y_pred = model.predict(test_inputs)
y_predi = y_pred

for i in range(TEST_LENGTH):
    f, axarr = plt.subplots(2, 5)

    axarr[1][4].imshow(test_inputs[i])
    axarr[1][4].set_title("original")

    axarr[1][3].imshow(onehot_to_rgb(test_labels[i], id2code))
    axarr[1][3].set_title("truth")

    axarr[1][2].imshow(onehot_to_rgb(y_pred[i], id2code))
    axarr[1][2].set_title("prediction mask")

    for cl in range(len(color_labels)):
        if cl > 5:
            axarr[1][cl - 5].imshow(y_pred[i, :, :, cl])
            axarr[1][cl - 5].set_title("layer " + id2code[cl + 1])
        else:
            axarr[0][cl].imshow(y_pred[i, :, :, cl])
            axarr[0][cl].set_title("layer " + id2code[cl + 1])

    plt.show()
