import os
import os.path as osp

import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from config import color_labels, data_set_path, id2code


def rgb_to_onehot(rgb_image, colormap):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2] + (num_classes,)
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 3)) == color_labels[colormap[cls]], axis=1).reshape(
            shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    single_layer = np.zeros((onehot.shape[:2]), dtype=np.uint8)
    for i in range(onehot.shape[2]):
        mask = np.round(onehot[:, :, i])
        mask = mask.astype(dtype=np.uint8)
        single_layer[mask == 1] = (i + 1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in colormap.keys():
        output[single_layer == k] = color_labels[colormap[k]]
    return np.uint8(output)


def prediction_to_rgb(pred, colormap):
    single_layer = np.argmax(pred, axis=-1)
    output = np.zeros(pred.shape[:2] + (3,))
    for k in colormap.keys():
        output[single_layer == k] = color_labels[colormap[k]]
    return np.uint8(output)


def generate_training_set(training_lenth, img_height, img_width, img_channels):
    masks_path = data_set_path + '/CelebA-HQ-img/'
    train_inputs = np.zeros((training_lenth, img_height, img_width, img_channels), dtype=np.uint8)
    for seq, _id in tqdm(enumerate(range(0, training_lenth)), total=training_lenth):
        train_filename = str(seq) + '.jpg'
        train_path = osp.join(masks_path, train_filename)
        train_input = imread(train_path)[:, :, :img_channels]
        train_input = resize(train_input, (img_height, img_width), mode='constant', preserve_range=True)
        train_inputs[seq] = train_input
    return train_inputs


def generate_labels(training_length, img_height, img_width):
    labels_length = len(color_labels)

    atts = color_labels.keys()

    output_images = np.zeros((training_length, img_height, img_width, labels_length), dtype=np.uint8)
    masks_path = data_set_path + '/CelebAMask-HQ-mask-anno/0/'
    for seq, _id in tqdm(enumerate(range(0, training_length)), total=training_length):
        masks = []

        for l, att in enumerate(atts, 1):
            file_name = masks_path + ''.join([str(seq).rjust(5, '0'), '_', att, '.png'])
            if os.path.exists(file_name):
                mask = np.array(Image.open(file_name).convert('P'))
                mask = resize(mask, (img_height, img_width), mode='constant', preserve_range=True)
                mask[mask == 225] = l
                masks.append(mask)

        coded_masks = np.array(masks)

        single_layer = np.max(coded_masks, axis=0, keepdims=False)
        output = np.zeros(coded_masks.shape[1:3] + (3,))

        for k in id2code.keys():
            output[single_layer == k] = color_labels[id2code[k]]

        output = np.uint8(output)

        encoded = rgb_to_onehot(output, id2code)
        output_images[seq] = encoded
    return output_images
