import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input, _preprocess_symbolic_input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Add, Dropout
from tensorflow.python.keras.models import load_model

from config import color_labels, id2code
from data_generator import generate_labels, generate_training_set, onehot_to_rgb, shuffle
from image_preprocessing import noisy, mean_filter, gaussian_blur
from metrics import dice, iou_coef
from tensorboard_callbacks import TensorBoardMask2


def create(n_classes=1, base=4, pretrained=False, pretrained_model_path='', learning_rate=1e-6, metrics=[dice]):
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

    i = Input(shape=INPUT_SHAPE)
    converted_inputs = tf.keras.layers.Lambda(lambda x: preprocess_input(x, mode='torch'))(i)

    ## Block 1
    x = Conv2D(2 ** base, (3, 3), activation='relu', padding='same', name='block1_conv1')(converted_inputs)
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=metrics)
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
TRAIN_LENGTH = 12000

train_inputs = generate_training_set(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
train_labels = generate_labels(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH)

# Image manipulations
train_inputs[:999] = [noisy(noise_type="gauss", image=image) for image in train_inputs[:999]]
train_inputs[1000:1999] = [noisy(noise_type="s&p", image=image) for image in train_inputs[1000:1999]]
train_inputs[2000:2999] = [noisy(noise_type="poisson", image=image) for image in train_inputs[2000:2999]]
train_inputs[3000:3999] = [noisy(noise_type="speckle", image=image) for image in train_inputs[3000:3999]]

train_inputs[4000:4999] = [mean_filter(image) for image in train_inputs[4000:4999]]
train_inputs[5000:5999] = [gaussian_blur(image) for image in train_inputs[5000:5999]]

# Shuffle
train_inputs, train_labels = shuffle(train_inputs, train_labels)

test_inputs = train_inputs[5:10]
test_labels = train_labels[5:10]

# Model
model = create(base=6, n_classes=len(color_labels), pretrained=False,
               pretrained_model_path='models/unet_20210515-134641.model',
               learning_rate=1e-5, metrics=[
        dice,
        'accuracy',
        iou_coef
    ])

# Model checkpoints
model_name = 'fcn8_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join('models', model_name + '.model'), verbose=1,
                                                save_best_only=True, mode='max',
                                                save_weights_only=False, period=10)

# Model callbacks
logdir = "logs/fit/" + model_name
callbacks = [
    checkpoint,
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
    TensorBoardMask2(original_images=test_inputs, log_dir=logdir, log_freq=5)
]

# Model learning
result = model.fit(train_inputs, train_labels, validation_split=0.2, batch_size=14, epochs=100, callbacks=callbacks)

model.save('models/' + model_name + '.model')

y_pred = model.predict(test_inputs)
y_predi = y_pred

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
