# configuration vars we want to set in one place

# imshape should have 3 channels for rgb input images
# (height, width)
imshape = (512, 512, 3)
# set your classification mode (binary or multi)
mode = 'binary'
# model_name (unet or fcn_8)
model_name = 'unet_' + mode

# classes are defined in hues
# background should be left out
color_labels = {'skin': (0, 120, 0),
                'cloth': (0, 240, 0),
                'hair': (240, 0, 60),
                'l_brow': (0, 0, 120),
                'l_eye': (0, 0, 180),
                'l_lip': (0, 0, 240),
                'mouth': (60, 0, 0),
                'neck': (120, 0, 0),
                'nose': (180, 0, 0),
                'r_brow': (240, 0, 0),
                'r_eye': (0, 60, 0),
                'u_lip': (0, 180, 0),
                'r_ear': (60, 60, 0),
                'l_ear': (120, 60, 0),
                'hat': (180, 60, 0),
                'eye_g': (240, 60, 0),
                'neck_l': (60, 120, 0),
                'ear_r': (60, 180, 0),
                }

hues = {'hair': 0,
        'l_brow': 10,
        'l_eye': 20,
        'l_lip': 30,
        'mouth': 40,
        'neck': 50,
        'nose': 60,
        'r_brow': 70,
        'r_eye': 80,
        'skin': 90,
        'u_lip': 100,
        'cloth': 110,
        'r_ear': 120,
        'l_ear': 130,
        'hat': 140,
        'eye_g': 150,
        'neck_l': 160,
        'ear_r': 170,
        }

labels = sorted(hues.keys())

id2code = {k + 1: v for k, v in enumerate(color_labels)}

if mode == 'binary':
    n_classes = 1

elif mode == 'multi':
    n_classes = len(labels) + 1

assert imshape[0] % 32 == 0 and imshape[1] % 32 == 0, \
    "imshape should be multiples of 32. comment out to test different imshapes."
