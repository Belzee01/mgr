import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from PIL import Image
import random

model = tf.keras.models.load_model('saved_model/my_model')
model.summary()

# Input dimensions
IMG_WIDTH = 512
IMG_HEIGHT = 512
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
    mask = np.zeros((512, 512))
    sep_mask = np.array(Image.open(mask_path).convert('P'))
    mask[sep_mask == 255] = 1
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                 preserve_range=True), axis=-1)
    test_labels[i] = mask

loss, acc = model.evaluate(test_inputs, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
preds_test = model.predict(test_inputs[int(test_inputs.shape[0] * 0.1):], verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
fig = plt.figure(figsize=(1, 3))
ix = random.randint(0, len(preds_test_t) - 1)
ax = fig.add_subplot(1, 3, 1)
ax.imshow(test_inputs[ix])
ax.set_title("original")

ax = fig.add_subplot(1, 3, 2)
ax.imshow(np.squeeze(test_labels[ix]))
ax.set_title("true class")

ax = fig.add_subplot(1, 3, 3)
ax.imshow(np.squeeze(preds_test_t[ix]))
ax.set_title("predicted class")

plt.show()
