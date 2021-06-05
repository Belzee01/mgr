import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

frame_number = -1

while(True):
    frame_number += 1
    ret, frame = cap.read()
    if frame_number == 3:  # if frame is the third frame than replace it with blank drame
      change_frame_with = np.zeros_like(frame)
      frame = change_frame_with
    out.write(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()