import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from config import color_labels


def rgb_to_onehot(rgb_image, colormap):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2] + (num_classes,)
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 3)) == color_labels[colormap[cls]], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    single_layer = np.nanargmax(onehot, axis=-1)
    skin_layer = np.nanargmin(onehot, axis=-1)
    single_layer[skin_layer == 1] = 1
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(single_layer)
    plt.show()
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in colormap.keys():
        output[single_layer == k] = color_labels[colormap[k]]
    return np.uint8(output)


DATA_SET_PATH = '/Users/klipensk/Documents/CelebAMask-HQ'

atts = ['skin', 'cloth', 'hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_eye',
        'u_lip', 'r_ear', 'l_ear', 'hat', 'eye_g', 'neck_l', 'ear_r']

MASKS_PATH = DATA_SET_PATH + '/CelebAMask-HQ-mask-anno/0/'

masks = []
id2code = {k + 1: v for k, v in enumerate(color_labels)}

for l, att in enumerate(atts, 1):
    file_name = MASKS_PATH + ''.join([str(52).rjust(5, '0'), '_', att, '.png'])
    if os.path.exists(file_name):
        mask = np.array(Image.open(file_name).convert('P'))
        mask[mask == 225] = l
        masks.append(mask)

coded_masks = np.array(masks)

single_layer = np.max(coded_masks, axis=0, keepdims=False)
output = np.zeros(coded_masks.shape[1:3] + (3,))

for k in id2code.keys():
    output[single_layer == k] = color_labels[id2code[k]]

output = np.uint8(output)

encoded = rgb_to_onehot(output, id2code)
print(encoded.shape)


decoded = onehot_to_rgb(encoded, id2code)
# fig = plt.figure(figsize=(4, 4))
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.imshow(decoded)
# plt.show()

