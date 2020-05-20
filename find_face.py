import cv2
import numpy as np

# Create a CascadeClassifier Object

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Reading the image as it is
# img = cv2.imread("C:\devl\photo.jpg")

# Reading the image as gray scale image
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Search the co-ordintes of the image
# faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
# for x, y, w, h in faces:
#   img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)

# resized = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

# cv2.imshow("Gray", resized)

# cv2.waitKey(0)
# gray_img, scaleFactor=1.05, minNeighbors=5
# cv2.destroyAllWindows()
# ----------------------------
video = cv2.VideoCapture("C:\devl\me.mp4")


a = 0
while True:
    a += 1
    cheeck, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for x, y, w, h in faces:
       img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 10)

    cv2.imshow("caps", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print(a)
video.release()
cv2.destroyAllWindows()
