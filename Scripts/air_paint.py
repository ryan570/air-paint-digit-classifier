import os
import sys
from datetime import datetime

import cv2
import numpy as np

from image_utils import *

CLASSIFY = True
SHOW_VIDEO = False
VIDEO_SIZE = (1080, 1920)
#I am using the IP Webcam android app to stream video to my pc. Use 0 for regular webcam 
VIDEO_SOURCE = 'http://192.168.1.134:8080/video'

if CLASSIFY:
    from model import predict

# empty callback for trackbar
def callback(x):
    pass

# start video capture
try:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
except:
    sys.exit()

# read off first frame and convert it to hsv
ret, frame = cap.read()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# get bbox for object to track
bbox = cv2.selectROI("Select Object", frame)
x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

# find mean and std hsv values for object to track
cropped = hsv[y:y+h, x:x+w]
mean, std = cv2.meanStdDev(cropped)
mean, std = [round(val) for sublist in mean for val in sublist], [
    round(val) for sublist in std for val in sublist]

# create window and trackbar to adjust # of standard deviations from the mean to include in mask
cv2.namedWindow("params")
cv2.createTrackbar("std", "params", 3, 5, callback)

# creates 2 blank black canvases. displayCanvas is what is displayed and has writing on it while canvas is fed to the CNN
canvas = np.zeros((*VIDEO_SIZE, 3), dtype=np.uint8)
displayCanvas = canvas.copy()

while True:
    # read each frame and convert it to HSV
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #displayCanvas must be copied to get rid of text written on it from the previous frame
    displayCanvas = canvas.copy()

    #compute lower and upper HSV color bounds and mask frame using them
    lower = np.subtract(
        mean, [val * cv2.getTrackbarPos("std", "params") for val in std])
    upper = np.add(
        mean,  [val * cv2.getTrackbarPos("std", "params") for val in std])
    masked = cv2.inRange(hsv, lower, upper)

    #find contours in masked image to determine where the object is
    contours, hierarchy = cv2.findContours(
        masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        #only paint on largest contour to eliminate residual noise
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.circle(canvas, (round(x + w/2), round(y + h/2)),
                       20, (255, 0, 0), cv2.FILLED)
            cv2.circle(displayCanvas, (round(x + w/2), round(y + h/2)),
                       20, (255, 0, 0), cv2.FILLED)
            
            #draw bbox around object on original video frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if CLASSIFY:
        #preprocess canvas, predict using CNN, and display prediction
        processed = preprocess(canvas)
        prediction = predict(processed)
        cv2.putText(displayCanvas, "You've drawn a(n) " + str(prediction),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    if SHOW_VIDEO:
        cv2.imshow("Video", frame)
    cv2.imshow("Masked", masked)
    cv2.imshow("Canvas", displayCanvas)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        #save canvas
        path = os.path.dirname(os.path.dirname(__file__)) + '/Drawings/'
        filename = datetime.now().strftime("%m-%d-%Y_%H-%M") + '.png'
        cv2.imwrite(os.path.join(path, filename), displayCanvas)
        break
    elif k == ord('c'):
        #clear canvas
        canvas = np.zeros((*VIDEO_SIZE, 3), dtype=np.uint8)
        displayCanvas = canvas.copy()
