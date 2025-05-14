import time
import numpy as np
import cv2 as cv
import mediapipe as mp


wCam, hCam = 640,360

cap = cv.VideoCapture(0)
#se funcionionar o while true é executado
if not cap.isOpened():
    print("camera não indentificada")
    exit()
while True:
    # Capture frame por frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

face_cascade  =  cv . CascadeClassifier ( 'haarcascade_frontalface_default.xml' ) 
eye_cascade  =  cv . CascadeClassifier ( 'haarcascade_eye.xml' )

img  =  cv . imread ( 'sachin.jpg' ) 
cinza  =  cv . cvtColor ( img ,  cv . COLOR_BGR2GRAY )

mpHands = mp.solutions.hands
Hands = mpHands.Hands(False)