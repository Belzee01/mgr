import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input, _preprocess_symbolic_input
from metrics import dice, mean_iou
from skimage.transform import resize
import numpy as np

model = load_model('models/deeplab_v3_20210525-183257.model',
                   custom_objects={'dice': dice, 'preprocess_input': preprocess_input,
                                   'mean_iou': mean_iou,
                                   '_preprocess_symbolic_input': _preprocess_symbolic_input,
                                   'tf.compat.v1.image.resize': tf.compat.v1.image.resize
                                   })
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice,
                                                                          'accuracy',
                                                                          mean_iou
                                                                          ])

# Input dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
ITEM_LENGTH = 2

cap = cv2.VideoCapture("C:\\Users\\Belzee\\Downloads\\Harry Styles - Watermelon Sugar (Official Video)\\Harry Styles - Watermelon Sugar.mp4")

if not cap.isOpened():
    print("Error opening video  file")

prediction_batch = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        resized_frame = resize(frame, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True)
        prediction_batch[0] = resized_frame
        prediction_mask = model.predict(prediction_batch, verbose=1)
        mask = resize(prediction_mask[0], (240, 426), preserve_range=True)
        resized_frame = resize(frame, (240, 426))

        cv2.imshow('Prediction', mask)
        cv2.imshow('Original', resized_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
