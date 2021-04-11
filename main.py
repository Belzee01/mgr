import os

import cv2
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from config import imshape, labels, hues

if __name__ == '__main__':
    # a = np.random.random((224, 224))
    # print(a)
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # plt.show()
    face_data = './CelebAMask-HQ/CelebA-HQ-img'
    face_sep_mask = './CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    mask_path = './CelebAMask-HQ/mask'
    blank = np.zeros(shape=imshape, dtype=np.uint8)

    for i, label in enumerate(labels):
        mask = np.zeros((512, 512))
        file_name = ''.join([str(0).rjust(5, '0'), '_', label, '.png'])
        path = osp.join(face_sep_mask, str(0), file_name)

        if os.path.exists(path):
            sep_mask = np.array(Image.open(path).convert('P'))
            hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
            sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
            val = sep_mask[:, :].astype(np.uint8)

            im_hsv = cv2.merge([hue, sat, val])
            im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
            blank = cv2.add(blank, im_rgb)

    plt.imshow(blank)
    plt.show()
