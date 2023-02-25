from unittest import result
import numpy as np
import cv2
import scipy
import math

from sort_leds import sort_leds
from calculate_angle import calulate_angle
from find_connections import find_connections
from process_img import process_img

from gradient import gradient_descent
from rotate import rotate_image

#
#.......................................................
VideoPath_1 = r"D:\John\Project\GOPRO\GOPRO_1\gopro_test_1.avi"
# VideoPath_2 = r"C:\Users\clare\Documents\John\Project\GOPRO\GOPRO_2"
# VideoPath_3 = r"C:\Users\clare\Documents\John\Project\GOPRO\GOPRO_3"
#       0   1   2
#           3      
#           4
#          
mm_wand_lenth_0_to_1 = 160
mm_wand_lenth_0_to_2 = 240
mm_wand_lenth_0_to_3 = 203
mm_wand_lenth_0_to_4 = 293
mm_wand_lenth_1_to_2 = 80
mm_wand_lenth_1_to_3 = 125
mm_wand_lenth_1_to_4 = 245
mm_wand_lenth_2_to_3 = 148
mm_wand_lenth_2_to_4 = 258
mm_wand_lenth_3_to_4 = 120


kernel = np.ones((5,5),np.uint8)
#.......................................................

#Stats
#.......................................................
cap = cv2.VideoCapture(VideoPath_1)
frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps= int(cap.get(cv2.CAP_PROP_FPS))
time = frames_count / fps
frame_progress = 0
#.......................................................




#Check if file is opened
if (cap.isOpened()== False):
    print("Error opening video")

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        frame_progress = frame_progress + 1
        if frame_progress > 321:
            cv2.imshow("original", img)
            #find LEDs
            closing, thresh1 = process_img(img, kernel)
            contours,h = cv2.findContours(closing,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

            #circular
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, .03 * cv2.arcLength(cnt, True), True)
               # print (len(approx))
                if len(approx)!=0:
                    area = cv2.contourArea(cnt)
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    circleArea = radius * radius * np.pi
                    cv2.drawContours(closing, [cnt], 0, (255, 0, 0), -1)
            bitand = cv2.bitwise_and(closing, thresh1)
            contours,h = cv2.findContours(bitand,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

            #process found led frame
            if len(contours) ==5:
                try:
                    cx, cy = find_connections (img, contours)  
                    cx, cy = sort_leds(cx, cy)
                    print("sorted")
                    calulate_angle(cx, cy)
                except Exception:
                    print("exception thrown")
                    pass
                else: 
                    cv2.line(img, (cx[0], cy[0]), (cx[1], cy[1]), (0, 255, 0), 4 )
                    # cv2.line(img, (cx[0], cy[0]), (cx[2], cy[2]), (0, 255, 0), 2 )
                    cv2.line(img, (cx[0], cy[0]), (cx[3], cy[3]), (0, 255, 0), 4 )
                    cv2.line(img, (cx[0], cy[0]), (cx[4], cy[4]), (0, 255, 0), 4 )
                    cv2.line(img, (cx[1], cy[1]), (cx[2], cy[2]), (0, 255, 0), 4 )
                    # cv2.line(img, (cx[1], cy[1]), (cx[4], cy[4]), (0, 255, 0), 2 )
                    cv2.line(img, (cx[2], cy[2]), (cx[4], cy[4]), (0, 255, 0), 4 )
                    cv2.line(img, (cx[2], cy[2]), (cx[3], cy[3]), (255, 0, 0), 4 )
                    cv2.line(img, (cx[1], cy[1]), (cx[3], cy[3]), (0, 0, 255), 2 )
                    cv2.line(img, (cx[3], cy[3]), (cx[4], cy[4]), (0, 0, 255), 2 )
                    cv2.imshow("contour", img)
                    # cv2.imshow("contour with original", cont_img)
                cv2.waitKey(0)

            cv2.putText(bitand, str(len(contours)), (20,20), cv2.FONT_HERSHEY_COMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            # cv2.imshow('contour',closing)
        #    cv2.imshow('combine',bitand)
            
        
        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Fin")
        break

    
# cv2.calibrateCamera(objpoints, imgpoints, cont_img.shape[::-1],None,None)


# When everything done, release
# the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

#Statistics
print("Total frames:      ", frames_count)
print("Frames per Second: ", fps)
print("Duration:          ", time, " seconds")