import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import color_labels, id2code
from data_generator import generate_training_set, generate_labels, onehot_to_rgb
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


seed = 42
np.random.seed = seed

# Input dimensions
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# Input data
TRAIN_LENGTH = 100
TEST_LENGTH = 2

train_inputs = generate_training_set(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
train_labels = generate_labels(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH)

test_inputs = train_inputs[5:10]
test_labels = train_labels[5:10]

# Model
model = U_Net.create(base=2, n_classes=len(color_labels))

# Model checkpoints
model_name = 'unet_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join('models', model_name + '.model'), verbose=1,
                                                save_best_only=True, mode='max',
                                                save_weights_only=False, period=10)

# Model callbacks
logdir = "logs/fit/" + model_name
callbacks = [
    checkpoint,
    tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
    TensorBoardMask2(original_images=test_inputs, log_dir=logdir, log_freq=5)
]

# Model learning
result = model.fit(train_inputs, train_labels, validation_split=0.1, batch_size=16, epochs=200, callbacks=callbacks)

model.save('models/' + model_name + '.model')

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
