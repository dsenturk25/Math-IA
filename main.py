
import cv2
import numpy as np

cap = cv2.VideoCapture('./images/video.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

height, width, _ = frame1.shape

res = np.zeros((height, width), np.uint8)

while cap.isOpened():

    kernel = np.ones((3,3), np.uint8)

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    res = cv2.bitwise_or(res, dilated)

    cv2.imshow('feed', res)
    cv2.imshow('real', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
