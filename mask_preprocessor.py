import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from config import imshape, labels, hues


def multi_class_image_loader(annotation_path, output_path):
    for label_num, not_used in enumerate(labels):
        for j in range(label_num * 2000, (label_num + 1) * 2000):
            blank = np.zeros(shape=imshape, dtype=np.uint8)
            for i, label in enumerate(labels):
                file_name = ''.join([str(j).rjust(5, '0'), '_', label, '.png'])
                path = osp.join(annotation_path, str(label_num), file_name)

                if os.path.exists(path):
                    sep_mask = np.array(Image.open(path).convert('P'))
                    hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
                    sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
                    val = sep_mask[:, :].astype(np.uint8)

                    im_hsv = cv2.merge([hue, sat, val])
                    im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
                    blank = cv2.add(blank, im_rgb)

            print('{}/{}.png'.format(output_path, j))
            cv2.imwrite('{}/{}.png'.format(output_path, j), blank)


def binary_class_image_loader(annotation_path, output_path):
    for i in range(15):
        atts = ['hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_eye',
                'skin', 'u_lip', 'cloth', 'r_ear', 'l_ear', 'hat', 'eye_g', 'neck_l', 'ear_r']
        for j in range(i * 2000, (i + 1) * 2000):
            mask = np.zeros((512, 512))
            for l, att in enumerate(atts, 1):
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(annotation_path, str(i), file_name)

                if os.path.exists(path):
                    sep_mask = np.array(Image.open(path).convert('P'))
                    # print(np.unique(sep_mask))
                    mask[sep_mask == 225] = 255
            cv2.imwrite('{}/{}.png'.format(output_path, j), mask)
            print('{}/{}.png'.format(output_path, j))


if __name__ == "__main__":
    face_sep_mask = './CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    mask_path = './CelebAMask-HQ/mask'
    multi_class_image_loader(face_sep_mask, mask_path)
