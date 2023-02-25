
import cv2


def find_connections (img, contours):
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
            raise Exception("Exception: led at corner")
                
  


            
    
    
    return cx, cy
  