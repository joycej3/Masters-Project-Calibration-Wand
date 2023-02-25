from unittest import result
import numpy as np
import cv2


def process_img(img, kernel):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2,thresh1 = cv2.threshold(grey,230,255,cv2.THRESH_BINARY)
    lower = np.array([155,150,150])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv,lower,upper)
    red = cv2.bitwise_and(hsv,hsv,mask=mask)
    red_to_grey = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)#test
    ret3,red_thresh = cv2.threshold(red_to_grey,0,255,cv2.THRESH_OTSU)
    open = cv2.morphologyEx(red_thresh, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)

# test
    return closing, thresh1


# def process_img(img, kernel):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret2,thresh1 = cv2.threshold(grey,230,255,cv2.THRESH_BINARY)
#     lower = np.array([155,150,150])
#     upper = np.array([179, 255, 255])
#     mask = cv2.inRange(hsv,lower,upper)
#     red = cv2.bitwise_and(hsv,hsv,mask=mask)
#     red_to_grey = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)#test
#     ret3,red_thresh = cv2.threshold(red_to_grey,0,255,cv2.THRESH_OTSU)
#     open = cv2.morphologyEx(red_thresh, cv2.MORPH_CLOSE, kernel)
#     closing = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)

#     return closing, thresh1