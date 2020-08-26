import cv2
import sys

lower = (18, 153, 179)
upper = (33, 255, 255)

def callback(val):
    pass

def get_trackbars(min):
    values = []

    for i in ["MIN", "MAX"]:
        for j in "HSV":
            values.append(cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars"))
    
    return values[:3] if min else values[3:]

try:
    cap = cv2.VideoCapture('http://192.168.1.134:8080/video')
except:
    sys.exit()

cv2.namedWindow("Trackbars")
for i in ["MIN", "MAX"]:
    v = 0 if i == "MIN" else 255

    for index, j in enumerate("HSV"):
        cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", lower[index] if i=="MIN" else upper[index], 255, callback)

while (cap.isOpened()):

    ret, img = cap.read()
    img = cv2.resize(img, (600, 350))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow("img", img)
    cv2.imshow("hsv", cv2.inRange(hsv, tuple(get_trackbars(True)), tuple(get_trackbars(False))))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("min", tuple(get_trackbars(True)))
        print("max", tuple(get_trackbars(False)))
        break