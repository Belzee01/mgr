# configuration vars we want to set in one place

# imshape should have 3 channels for rgb input images
# (height, width)
imshape = (256, 256, 3)
# set your classification mode (binary or multi)
mode = 'binary'
# model_name (unet or fcn_8)
model_name = 'unet_' + mode

# classes are defined in hues
# background should be left out
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

if mode == 'binary':
    n_classes = 1

elif mode == 'multi':
    n_classes = len(labels) + 1

assert imshape[0] % 32 == 0 and imshape[1] % 32 == 0, \
    "imshape should be multiples of 32. comment out to test different imshapes."
