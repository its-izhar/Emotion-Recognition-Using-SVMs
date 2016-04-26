import cv2
import numpy as np
from scipy.ndimage import zoom

def loadImage():
    img = cv2.imread("../Data/Test3.jpg")
    return img


def detectFaces():
    img = loadImage()
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
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255,0,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        horizontal_offset = int(0.15 * w)
        vertical_offset = int(0.2 * h)

    extracted_face = gray[y+vertical_offset : y+h, x+horizontal_offset : x-horizontal_offset+w]

    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],
                                               64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())

    #print new_extracted_face.shape
    cv2.imshow('New Extracted', new_extracted_face)
    cv2.imshow('Extracted', extracted_face)
    cv2.imshow('gray', roi_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return new_extracted_face.reshape(1,-1)
