# configuration vars we want to set in one place

# imshape should have 3 channels for rgb input images
# (height, width)
imshape = (512, 512, 3)
# set your classification mode (binary or multi)
mode = 'binary'
# model_name (unet or fcn_8)
model_name = 'unet_' + mode

data_set_path = '/Users/klipensk/Documents/CelebAMask-HQ'

# classes are defined in hues
# background should be left out
color_labels = {'skin': (0, 120, 0),
                'hair': (240, 0, 60),
                'mouth': (60, 0, 0),
                'neck': (120, 0, 0),
                'nose': (180, 0, 0),
                }

id2code = {k + 1: v for k, v in enumerate(color_labels)}

assert imshape[0] % 32 == 0 and imshape[1] % 32 == 0, \
    "imshape should be multiples of 32. comment out to test different imshapes."
