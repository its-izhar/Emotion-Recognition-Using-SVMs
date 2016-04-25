import cv2

img = cv2.imread("../Data/Test.jpg")

cascPath = "../data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

for (x,y,w,h) in detected_faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

