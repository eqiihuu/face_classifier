#! /usr/bin/python
# This script will detect faces via your webcam.
# Tested with OpenCV3

import numpy as np
import cv2
import copy
import time
from matplotlib import pyplot as plt
from keras.models import load_model

expression_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

img_dict = {}
for i in range(7):
    img_dict[i] = cv2.imread("./emoji2/"+expression_dict[i]+".png")

classifier = load_model('./cnn.h5')
cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    # Capture frame-by-frame
    t1 = time.time()
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    frame0 = copy.deepcopy(frame)
    for (x, y, w, h) in faces:
        try:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box = frame[y+1:y+h-1, x+1:x+w-1, :]
            box_gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            box_48 = cv2.resize(box_gray, (48, 48), interpolation=cv2.INTER_LINEAR)
            box_in = box_48.reshape((1, 48, 48, 1))
            expression = np.argmax(classifier.predict(box_in, batch_size=1))
            img = img_dict[expression]
            print expression_dict[expression]
            img = cv2.resize(img, (w, h))
            frame[y:y+h, x:x+w, :] = img
        except Exception:
            print "error"
    frame = np.concatenate((frame, frame0), axis=0)
    frame = cv2.resize(frame, (640, 720))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    t2 = time.time()
    print('Time: %.3f' % (t2-t1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



