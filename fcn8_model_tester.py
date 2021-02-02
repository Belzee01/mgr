import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

model = tf.keras.models.load_model('saved_model/fcn8')
model.summary()


def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return (seg_img)


# Input dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
ITEM_LENGTH = 3

# Load test data
face_data = './CelebAMask-HQ/CelebA-HQ-img'
mask_data = './CelebAMask-HQ/mask'
TEST_LENGTH = 2
TEST_OFFSET = 4004
test_inputs = np.zeros((TEST_LENGTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
test_labels = np.zeros((TEST_LENGTH, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

for i, id_ in tqdm(enumerate(range(TEST_OFFSET, TEST_OFFSET + TEST_LENGTH)), total=TEST_LENGTH):
    train_filename = str(i) + '.jpg'
    mask_filename = str(i) + '.png'
    train_path = osp.join(face_data, train_filename)
    mask_path = osp.join(mask_data, mask_filename)
    input = imread(train_path)[:, :, :IMG_CHANNELS]
    input = resize(input, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    test_inputs[i] = input
    # Set singular label
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    sep_mask = np.array(Image.open(mask_path).convert('P'))
    sep_mask = resize(sep_mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask[sep_mask == 255] = 1
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                 preserve_range=True), axis=-1)
    test_labels[i] = mask

loss, acc = model.evaluate(test_inputs, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

y_pred = model.predict(test_inputs)
y_predi = np.argmax(y_pred, axis=3)

shape = (224, 224)
for i in range(TEST_LENGTH):
    img_is = (test_inputs[i])
    seg = y_predi[i]

    fig = plt.figure(figsize=(1, 3))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_is)
    ax.set_title("original")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(give_color_to_seg_img(seg, 1))
    ax.set_title("predicted class")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(np.squeeze(test_labels[i]))
    ax.set_title("true class")
    plt.show()
