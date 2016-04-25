import cv2
import numpy as np
from scipy.ndimage import zoom

img = cv2.imread("../Data/Test.jpg")

cascPath = "../data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in detected_faces:
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255,0,255), 3)
    roi_gray = gray[y:y+h, x:x+w]
    horizontal_offset = int(0.15 * w)
    vertical_offset = int(0.2 * h)
    extracted_face = gray[y+vertical_offset : y+h, x+horizontal_offset : x-horizontal_offset+w]

cv2.imshow('Extracted', extracted_face)
cv2.imshow('gray', roi_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
