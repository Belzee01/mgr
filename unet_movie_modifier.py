import cv2
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input, _preprocess_symbolic_input
from metrics import dice, iou_coef
from skimage.transform import resize
import numpy as np

model = load_model('models/unet_20210530-135541.model',
                   custom_objects={'dice': dice, 'iou_coef': iou_coef, 'preprocess_input': preprocess_input,
                                   '_preprocess_symbolic_input': _preprocess_symbolic_input
                                   })
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
    dice,
    'accuracy',
    iou_coef
])

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

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
