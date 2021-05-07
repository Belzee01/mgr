import datetime
import os
from typing import Tuple
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Add, Dropout, BatchNormalization, \
    Activation, GlobalAveragePooling2D, Lambda, ZeroPadding2D, Concatenate, DepthwiseConv2D

from config import color_labels, id2code
from data_generator import generate_labels, generate_training_set, onehot_to_rgb
from metrics import dice
from tensorboard_callbacks import TensorBoardMask2

from tensorflow.python.keras import layers
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input


class Deeplabv3:
    @staticmethod
    def preprocess_input(x):
        return preprocess_input(x, mode='tf')

    @staticmethod
    def create(input_shape: Tuple[int, int, int], n_classes=1, base=4):
        if n_classes == 1:
            loss = 'binary_crossentropy'
            final_act = 'sigmoid'
        elif n_classes > 1:
            loss = 'categorical_crossentropy'
            final_act = 'softmax'

        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        i = Input(shape=input_shape)
        s = tf.keras.layers.Lambda(Deeplabv3.preprocess_input)(i)
        x = Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1', use_bias=False, padding='same')(i)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation(tf.nn.relu)(x)

        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False, dilation_rate=(1, 1),
                   name='entry_flow_conv1_2')(x)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation(tf.nn.relu)(x)

        x = Deeplabv3.xception_block(x, [128, 128, 128], 'entry_flow_block1', skip_connection_type='conv', stride=2,
                                     depth_activation=False)
        x, skip1 = Deeplabv3.xception_block(x, [256, 256, 256], 'entry_flow_block2', skip_connection_type='conv', stride=2,
                                            depth_activation=False, return_skip=True)

        x = Deeplabv3.xception_block(x, [728, 728, 728], 'entry_flow_block3',
                                     skip_connection_type='conv', stride=entry_block3_stride,
                                     depth_activation=False)
        for i in range(16):
            x = Deeplabv3.xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                         skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                         depth_activation=False)

        x = Deeplabv3.xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                                     skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                                     depth_activation=False)
        x = Deeplabv3.xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                                     skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                                     depth_activation=True)

        # end of feature extractor

        # branching for Atrous Spatial Pyramid Pooling
        # Image Feature branch
        b4 = GlobalAveragePooling2D()(x)

        # from (b_size, channels)->(b_size, 1, 1, channels)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation(tf.nn.relu)(b4)

        # upsample. have to use compat because of the option align_corners
        size_before = tf.keras.backend.int_shape(x)
        b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                        method='bilinear', align_corners=True))(b4)
        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)

        # rate = 6 (12)
        b1 = Deeplabv3.SepConv_BN(x, 256, 'aspp1', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = Deeplabv3.SepConv_BN(x, 256, 'aspp2', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = Deeplabv3.SepConv_BN(x, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])

        x = Conv2D(256, (1, 1), padding='same',
                   use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation(tf.nn.relu)(x)
        x = Dropout(0.1)(x)

        # DeepLab v.3+ decoder
        # Feature projection
        # x4 (x2) block
        skip_size = tf.keras.backend.int_shape(skip1)
        x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, skip_size[1:3], method='bilinear', align_corners=True))(x)

        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = Deeplabv3.SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = Deeplabv3.SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

        x = Conv2D(n_classes, (1, 1), padding='same', name="last layer")(x)
        size_before3 = tf.keras.backend.int_shape(i)
        x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, size_before3[1:3], method='bilinear', align_corners=True))(
            x)
        # Oytputs
        o = tf.keras.layers.Activation(final_act)(x)

        model = Model(i, o, name='deeplabv3plus')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=[dice, "accuracy"])
        model.summary()

        return model

    @staticmethod
    def xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                       rate=1, depth_activation=False, return_skip=False):
        residual = inputs
        for i in range(3):
            residual = Deeplabv3.SepConv_BN(residual,
                                  depth_list[i],
                                  prefix + '_separable_conv{}'.format(i + 1),
                                  stride=stride if i == 2 else 1,
                                  rate=rate,
                                  depth_activation=depth_activation)
            if i == 1:
                skip = residual
        if skip_connection_type == 'conv':
            shortcut = Deeplabv3.conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                              kernel_size=1,
                                              stride=stride)
            shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
            outputs = layers.add([residual, shortcut])
        elif skip_connection_type == 'sum':
            outputs = layers.add([residual, inputs])
        elif skip_connection_type == 'none':
            outputs = residual
        if return_skip:
            return outputs, skip
        else:
            return outputs

    @staticmethod
    def conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
        if stride == 1:
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='same', use_bias=False,
                          dilation_rate=(rate, rate),
                          name=prefix)(x)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='valid', use_bias=False,
                          dilation_rate=(rate, rate),
                          name=prefix)(x)

    @staticmethod
    def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        if not depth_activation:
            x = Activation(tf.nn.relu)(x)
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = Activation(tf.nn.relu)(x)
        x = Conv2D(filters, (1, 1), padding='same',
                   use_bias=False, name=prefix + '_pointwise')(x)
        x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = Activation(tf.nn.relu)(x)

        return x


seed = 42
np.random.seed = seed

# Input dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
TEST_LENGTH = 2

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# Input data
TRAIN_LENGTH = 100

train_inputs = generate_training_set(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
train_labels = generate_labels(TRAIN_LENGTH, IMG_HEIGHT, IMG_WIDTH)

test_inputs = train_inputs[5:10]
test_labels = train_labels[5:10]

# Model
model = FCN_8.create(input_shape=INPUT_SHAPE, base=6, n_classes=len(color_labels))

# Model checkpoints
model_name = 'fcn8_elu_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
result = model.fit(train_inputs, train_labels, validation_split=0.1, batch_size=32, epochs=100, callbacks=callbacks)

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
