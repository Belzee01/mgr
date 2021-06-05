import cv2

cap = cv2.VideoCapture("C:\\Users\\Belzee\\Downloads\\Dua Lipa - Love Again.mp4")

if not cap.isOpened():
    print("Error opening video  file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
