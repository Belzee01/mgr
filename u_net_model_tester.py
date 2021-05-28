import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input, _preprocess_symbolic_input

from config import id2code
from data_generator import generate_training_set, generate_labels, onehot_to_rgb
from metrics import dice, iou_coef

model = load_model('models/unet_20210523-140202.model',
                   custom_objects={'dice': dice, 'mean_iou': iou_coef, 'preprocess_input': preprocess_input,
                                   '_preprocess_symbolic_input': _preprocess_symbolic_input
                                   })
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
    dice,
    'accuracy',
    iou_coef
])

# Input dimensions
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
ITEM_LENGTH = 2

# Load test data
images = generate_training_set(ITEM_LENGTH, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
labels = generate_labels(ITEM_LENGTH, IMG_WIDTH, IMG_HEIGHT)

loss, dice, acc, iou_coef = model.evaluate(images, labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
preds_test = model.predict(images, verbose=1)
label = labels[0]
pred_label = preds_test[0]

f, axarr = plt.subplots(2, 3)

axarr[1][2].imshow(images[0])
axarr[1][2].set_title("original")

axarr[1][1].imshow(onehot_to_rgb(label, id2code))
axarr[1][1].set_title("truth")

axarr[1][0].imshow(onehot_to_rgb(pred_label, id2code))
axarr[1][0].set_title("prediction mask")

for i in range(label.shape[2]):
    if i > 3:
        axarr[1][i - 3].imshow(pred_label[:, :, i])
        axarr[1][i - 3].set_title("layer " + id2code[i + 1])
    else:
        axarr[0][i].imshow(pred_label[:, :, i])
        axarr[0][i].set_title("layer " + id2code[i + 1])

plt.show()
