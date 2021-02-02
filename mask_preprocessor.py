import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image

if __name__ == "__main__":
    face_data = './CelebAMask-HQ/CelebA-HQ-img'
    face_sep_mask = './CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    mask_path = './CelebAMask-HQ/mask'
    counter = 0
    total = 0
    for i in range(15):
        # files = os.listdir(osp.join(face_sep_mask, str(i)))

        # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        #         'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        for j in range(i * 2000, (i + 1) * 2000):

            mask = np.zeros((512, 512))

            for l, att in enumerate(atts, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name)

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))
                    # print(np.unique(sep_mask))

                    mask[sep_mask == 255] = 255
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            print('{}/{}.png'.format(mask_path, j))

    print(counter, total)
