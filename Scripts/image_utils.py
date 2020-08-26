import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_to_roi(img, padding=50):
    #find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #loop through contours to find smallest bbox that covers all contours
    minX, minY, maxX, maxY = img.shape[1], img.shape[0], 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        minX, minY = min(minX, x), min(minY, y)
        maxX, maxY = max(maxX, x+w), max(maxY, y+h)

    #crop img to bbox and add black padding to edges
    img = img[minY:maxY, minX:maxX]
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return img

def preprocess(img):
    #convert to grayscale and crop to roi
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cropped = crop_to_roi(img)

    #resize to 28x28 with 2 pixels of padding on each edge giving 32x32
    small = cv2.resize(cropped, (28, 28))
    small = np.pad(small, ((2, 2), (2, 2)))

    #add dim to agree with model input size
    small = np.expand_dims(small, -1)

    return small