from unittest import result
import numpy as np
import cv2
import pandas as pd
import openpyxl
import scipy
import math


#       4   1   0
#           2       
#           3
#          
objpoints = []
imgpoints = []
grey = 1



mm_wand_lenth_0_to_1 = 80
mm_wand_lenth_0_to_2 = 148
mm_wand_lenth_0_to_3 = 258
mm_wand_lenth_0_to_4 = 240
mm_wand_lenth_1_to_2 = 125
mm_wand_lenth_1_to_3 = 245
mm_wand_lenth_1_to_4 = 160
mm_wand_lenth_2_to_3 = 120
mm_wand_lenth_2_to_4 = 203
mm_wand_lenth_3_to_4 = 293


def find_contours (contours):
    i = 0
    cx = [None] * 5
    cy = [None] * 5
    for cnt in contours:

        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx[i] = int(M['m10']/M['m00'])
            cy[i] = int(M['m01']/M['m00'])
            # print(cx[i], ",  ", cy[i], "\n")
            i = i + 1
        else :
            raise Exception("led at corner")
                
  
    cv2.line(img, (cx[0], cy[0]), (cx[1], cy[1]), (0, 255, 0), 5 )
    cv2.line(img, (cx[0], cy[0]), (cx[2], cy[2]), (255, 0, 0), 5 )
    cv2.line(img, (cx[0], cy[0]), (cx[3], cy[3]), (0, 255, 0), 5 )
    cv2.line(img, (cx[0], cy[0]), (cx[4], cy[4]), (0, 255, 0), 5 )
    cv2.line(img, (cx[1], cy[1]), (cx[2], cy[2]), (0, 0, 255), 5 )
    cv2.line(img, (cx[1], cy[1]), (cx[3], cy[3]), (0, 0, 255), 5 )
    cv2.line(img, (cx[1], cy[1]), (cx[4], cy[4]), (0, 255, 0), 5 )
    cv2.line(img, (cx[2], cy[2]), (cx[3], cy[3]), (0, 0, 255), 5 )
    cv2.line(img, (cx[2], cy[2]), (cx[4], cy[4]), (0, 255, 0), 5 )
    cv2.line(img, (cx[3], cy[3]), (cx[4], cy[4]), (0, 255, 0), 5 )

            
    cx, cy = sort_leds(cx, cy)
    calulate_angle(cx, cy)
    return img
    

#sortes the cx and cy arrays by callibration wand
def sort_leds(cx, cy):
    min_max_coord = [None] * 4
    sorted_cx = [None] * 5
    sorted_cy = [None] * 5

    #which coordinates are on the outside
    
    min_max_coord[0] = np.argmin(cx)
    min_max_coord[1] = np.argmax(cx)
    min_max_coord[2] = np.argmin(cy)
    min_max_coord[3] = np.argmax(cy)

    #points that are not on the outside
    mid_points =[] #points that are in the midel of the bounding box
    for x in range(0, 5):
        if x not in min_max_coord:
            mid_points.append(x)
            
  
    #if the middle point is known (unlikely)
    if len(mid_points) == 1:
        print(" error one found")
        

    elif len(mid_points) == 2 :        
        linex = [cx[mid_points[0]], cx[mid_points[1]]]  #two cx points
        liney = [cy[mid_points[0]], cy[mid_points[1]]]  #two cy points
        linex.append(None)
        liney.append(None)
        max_r = -2 #value is between -1 and +1
        for x in range(0, 5):
            if x in min_max_coord:
                linex[2] = cx[x]
                liney[2] = cy[x]
                pearsons_r = scipy.stats.pearsonr(linex, liney)[0]
                #new max
                if pearsons_r > max_r:
                    max_r = pearsons_r
                    #found botttom led
                    sorted_cx[4] = cx[x]
                    sorted_cy[4] = cy[x]

        #found #4 now find others 
        distance1 = math.dist( (sorted_cx[4], sorted_cy[4]) , (cx[mid_points[0]], cy[mid_points[0]]  ))
        distance2 = math.dist( (sorted_cx[4], sorted_cy[4]) , (cx[mid_points[1]], cy[mid_points[1]]  ))   
        if distance1 < distance2:
            sorted_cx[3] = cx[mid_points[0]]
            sorted_cy[3] = cy[mid_points[0]]
            sorted_cx[1] = cx[mid_points[1]]
            sorted_cy[1] = cy[mid_points[1]]
        else:
            sorted_cx[1] = cx[mid_points[0]]
            sorted_cy[1] = cy[mid_points[0]]
            sorted_cx[3] = cx[mid_points[1]]
            sorted_cy[3] = cy[mid_points[1]]

        #found 3/5 LEDs
        #find led 0 and 2
        for x in range(0,5):
            if cx[x] not in sorted_cx and cy[x] not in sorted_cy:

                if sorted_cx[4] == sorted_cx[1]:
                    print("Error coordinates are incorrect")

                elif sorted_cx[4] < sorted_cx[1]:
                    #0 is above
                    if sorted_cy[1] < cy[x]:
                        sorted_cx[0] = cx[x]
                        sorted_cy[0] = cy[x]
                    elif sorted_cy[1] > cy[x]:
                        sorted_cx[2] = cx[x]
                        sorted_cy[2] = cy[x]
                    else:
                        if sorted_cx[1] > cx[x]:
                            sorted_cx[0] = cx[x]
                            sorted_cy[0] = cy[x]
                        else:
                            sorted_cx[2] = cx[x]
                            sorted_cy[2] = cy[x]
                else:
                    #0 is below
                   
                    if sorted_cy[1] < cy[x]:
                        sorted_cx[2] = cx[x]
                        sorted_cy[2] = cy[x]
                    else:
                        sorted_cx[0] = cx[x]
                        sorted_cy[0] = cy[x]

    else:
        print("Error finding points\n")  

    return sorted_cx, sorted_cy

def calulate_angle(cx, cy):
    #pixel distances
    pixel_wand_lenth_0_to_1= math.dist( (cx[0], cy[0]), (cx[1], cy[1]))
    pixel_wand_lenth_0_to_2= math.dist( (cx[0], cy[0]), (cx[2], cy[2]))
    pixel_wand_lenth_0_to_3= math.dist( (cx[0], cy[0]), (cx[3], cy[3]))
    pixel_wand_lenth_0_to_4= math.dist( (cx[0], cy[0]), (cx[4], cy[4]))
    pixel_wand_lenth_1_to_2= math.dist( (cx[1], cy[1]), (cx[2], cy[2]))
    pixel_wand_lenth_1_to_3= math.dist( (cx[1], cy[1]), (cx[3], cy[3]))
    pixel_wand_lenth_1_to_4= math.dist( (cx[1], cy[1]), (cx[4], cy[4]))
    pixel_wand_lenth_2_to_3= math.dist( (cx[2], cy[2]), (cx[3], cy[3]))
    pixel_wand_lenth_2_to_4= math.dist( (cx[2], cy[2]), (cx[4], cy[4]))
    pixel_wand_lenth_3_to_4= math.dist( (cx[3], cy[3]), (cx[4], cy[4]))

    #world dimensions
    #pitch, yaw, roll
    #pitch

    #chech upside down
    if cy[3] < cy[1]: 
        print(cy[3], "  ", cy[1])
        print("upside down")
    else:
        print("right way")

    if cx[3] < cx[1]: 
        print(cx[3], "  ", cx[1])
        print("facing right")
    else:
        print("facing left")

    diff_cx = cx[1] - cx[3]
    diff_cy = cy[1] - cy[3]   
    handle_angle = math.atan(diff_cy/diff_cx)
    #print(handle_angle)

    diff_cx = cx[4] - cx[1]
    diff_cy = cy[4] - cy[1]   
    head_angle = math.atan(diff_cy/diff_cx)
  # print(head_angle)
    print((head_angle + handle_angle)/2)
    print("......................")

    print (pixel_wand_lenth_0_to_1 / mm_wand_lenth_0_to_1)
    print (pixel_wand_lenth_0_to_2 / mm_wand_lenth_0_to_2)
    print (pixel_wand_lenth_0_to_3 / mm_wand_lenth_0_to_3)
    print (pixel_wand_lenth_0_to_4 / mm_wand_lenth_0_to_4)
    print (pixel_wand_lenth_1_to_2 / mm_wand_lenth_1_to_2)
    print (pixel_wand_lenth_1_to_3 / mm_wand_lenth_1_to_3)
    print (pixel_wand_lenth_1_to_4 / mm_wand_lenth_1_to_4)
    print (pixel_wand_lenth_2_to_3 / mm_wand_lenth_2_to_3)
    print (pixel_wand_lenth_2_to_4 / mm_wand_lenth_2_to_4)
    print (pixel_wand_lenth_3_to_4 / mm_wand_lenth_3_to_4)

    # print( mm_wand_lenth_3_to_4 / mm_wand_lenth_1_to_3)

    
    return 


# xlpath = 'pandas_to_excel.xlsx'
# df = pd.DataFrame([[12, 21, 31], [12, 22, 32], [31, 32, 33]],
#                   index=['one', 'two', 'three'], columns=['a', 'b', 'c'])
# df.to_excel('pandas_to_excel.xlsx', sheet_name='new_sheet_name')

VideoPath_1 = r"D:\John\Project\GOPRO\GOPRO_1\gopro_test_1.avi"
# VideoPath_2 = r"C:\Users\clare\Documents\John\Project\GOPRO\GOPRO_2"
# VideoPath_3 = r"C:\Users\clare\Documents\John\Project\GOPRO\GOPRO_3"

cap = cv2.VideoCapture(VideoPath_1)
kernel = np.ones((5,5),np.uint8)

#Stats
frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps= int(cap.get(cv2.CAP_PROP_FPS))
time = frames_count / fps
frame_progress = 0

#Check if file is opened
if (cap.isOpened()== False):
    print("Error opening video")

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        frame_progress = frame_progress + 1
        if frame_progress > 321:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #grey test
            cv2.imshow("original", img)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret2,thresh1 = cv2.threshold(grey,230,255,cv2.THRESH_BINARY)


            #working 
            lower = np.array([155,150,150])
            upper = np.array([179, 255, 255])
            mask = cv2.inRange(hsv,lower,upper)
            red = cv2.bitwise_and(hsv,hsv,mask=mask)
            red_to_grey = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)#test
           # cv2.putText(red, str(frame_progress), (20,20), cv2.FONT_HERSHEY_COMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            ret3,red_thresh = cv2.threshold(red_to_grey,0,255,cv2.THRESH_OTSU)
            open = cv2.morphologyEx(red_thresh, cv2.MORPH_CLOSE, kernel)

            closing = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
            
            
            #contours
            contours,h = cv2.findContours(closing,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
          #  cv2.putText(closing, str(len(contours)), (20,20), cv2.FONT_HERSHEY_COMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            
            for cnt in contours:

                approx = cv2.approxPolyDP(cnt, .03 * cv2.arcLength(cnt, True), True)
               # print (len(approx))
                if len(approx)!=0:
                    area = cv2.contourArea(cnt)
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    circleArea = radius * radius * np.pi
                   # print (circleArea)
                  #  print (area)
                   # if circleArea == area:
                    cv2.drawContours(closing, [cnt], 0, (255, 0, 0), -1)
            bitand = cv2.bitwise_and(closing, thresh1)

            contours,h = cv2.findContours(bitand,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            

           
            if len(contours) ==5:
                try:
                    cont_img = find_contours (contours)
                except Exception:
                    pass
                else: 
                    cv2.imshow("contour with original", cont_img)
                cv2.waitKey(0)

            cv2.putText(bitand, str(len(contours)), (20,20), cv2.FONT_HERSHEY_COMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
          #  cv2.imshow('contour',closing)
           # cv2.imshow('combine',bitand)
            
        
        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Fin")
        break

    
cv2.calibrateCamera(objpoints, imgpoints, cont_img.shape[::-1],None,None)


# When everything done, release
# the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

#Statistics
print("Total frames:      ", frames_count)
print("Frames per Second: ", fps)
print("Duration:          ", time, " seconds")