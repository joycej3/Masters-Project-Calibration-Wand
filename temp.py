from matplotlib import pyplot as plt
import numpy as np
import cv2

from sort_leds import sort_leds
from calculate_angle import calulate_angle
from process_img import find_connections
from rotate import rotate_image

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
  
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)
  
VideoPath_1 = r"D:\John\Project\Trial1\Cal 02.2144931.20220906113403.avi"
template = cv2.imread('D:\John\Project\Code\masters-project\wtemplate.png') 

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

#Stats
cap = cv2.VideoCapture(VideoPath_1)
frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps= int(cap.get(cv2.CAP_PROP_FPS))
time = frames_count /fps
frame_progress = 0
#.......................................................

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
#Check if file is opened
if (cap.isOpened()== False):
    print("Error opening video")
# print("0")
while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        frame_progress = frame_progress + 1
        if frame_progress > 400:
            img_height, img_width = img.shape[:2]
            blank_temp = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            led_pixels = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            og = img.copy()

            fgmask = fgbg.apply(img)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 220])
            upper = np.array([255, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_and(mask, fgmask)

            saturated_pixels = cv2.bitwise_and(img, img, mask=mask)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([20, 255, 255])    
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            kernel = np.ones((3,3),np.uint8)
            red_mask = cv2.dilate(red_mask, kernel, iterations=1)
            mask = cv2.bitwise_and(fgmask, mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)        

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            led_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter != 0:
                    circularity = 4 * np.pi * area / perimeter**2
                else:
                    circularity = 0 
                if area > 10 and area < 100 and circularity > 0.25:
                    # print(contour)
                    led_contours.append(contour)

            # Draw the contours on the original image
            cv2.drawContours(img, led_contours, -1, (255, 0, 0), -1)

            if len(led_contours) !=6546:
                cv2.drawContours(led_pixels, led_contours, -1, (255, 255, 255), -1)
                cv2.imshow('image', og)
                cv2.imshow("cont", led_pixels)
                cv2.setMouseCallback('cont', click_event)
                gray_mask = cv2.cvtColor(led_pixels, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                # gray_img_dilated = cv2.cvtColor(img_dilated, cv2.COLOR_BGR2GRAY)
                # ..........................................................

                # # Initialize the SIFT detector
                # sift = cv2.SIFT_create()


                # Detect keypoints and compute descriptors
                # kp, des = sift.detectAndCompute(gray_img, None)
                # print(f"des data type: {des.dtype}")
                # print(f"des shape: {des.shape}")
                # kp2, des2 = sift.detectAndCompute(blank_temp, None)
                # print(f"des2 data type: {des2.dtype}")
                # print(f"des2 shape: {des2.shape}")


                # bf = cv2.BFMatcher()
                # matches = bf.match(des, des2)

                # # Sort matches by distance
                # matches = sorted(matches, key=lambda x: x.distance)

                # # # Apply ratio test to eliminate weak matches
                # good_matches = []
                # for m in matches:
                #     if m.distance > 0.75 * matches[0].distance:
                #         good_matches.append(m)
                # # # if good_matches:
                # cv2.imshow("original", og)
                # img3 = cv2.drawMatches(gray_img, kp, blank_temp, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # cv2.imshow("Matches", img3)
                
                # Display result
                # Draw matches on image
                
                
                # cv2.waitKey(0)


                 # ............................................................................................
                # try:
                # cx, cy = find_connections (img, led_contours)  
                # cy = [None] * 5
                # cx = [None] * 5
                # cx[0] = 495
                # cx[1] = 436
                # cx[2] = 412
                # cx[3] = 461
                # cx[4] = 482

                # cx[0] = 474
                # cx[1] = 443
                # cx[2] = 426
                # cx[3] = 396
                # cx[4] = 353
                # # img_points = np.array([[495, 474], [436, 443], [412, 426], [461, 396], [482, 353]], dtype=np.float32)
                # # real_points1 = np.float32([[0, 0,0 ], [160, 0,0], [240, 0,0], [160, 80,0], [160, 245,0]])
                # img_points = []
                # obj_points = []
                # camera_matrix = np.zeros((3, 3))
                # dist_coeffs = np.zeros((5, 1))

                # image_points = np.array([[495, 474], [436, 443], [412, 426], [461, 396], [482, 353]], dtype=np.float32)
                # img_points.append(image_points)  
                # # The focal length of the camera is 307.8219886231129 pixels
                # # The focal length of the camera is 547.239090885534 mm
            
                # image_points = np.array([[235, 436], [195, 367], [175, 335], [251, 332], [311, 291]], dtype=np.float32)
                # img_points.append(image_points)  
                # # The focal length of the camera is 115.55118389395169 pixels
                # # The focal length of the camera is 205.42432692258078 mm

                # image_points = np.array([[1021, 127], [1007, 183], [1001, 215], [964, 188], [917, 196]], dtype=np.float32)
                # img_points.append(image_points)  

                # image_points = np.array([[1000, 336], [962, 3773], [936, 402], [920, 354], [877, 335]], dtype=np.float32)
                # img_points.append(image_points)  
 


                # object_points = np.array([[0, 0,0 ], [160, 0,0], [240, 0,0], [160, 80,0], [160, 245,0]], dtype=np.float32)
                # obj_points.append(object_points)
                # obj_points.append(object_points)
                # obj_points.append(object_points)
                # obj_points.append(object_points)
                # retval, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (img_height, img_width), None, None)      
            

                # # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
                # focal_length_px = cameraMatrix[0, 0]
                # focal_length_mm = focal_length_px * (img_width/img_height)
                # # print(f"The focal length of the camera is {focal_length_px} pixels")
                # # print(f"The focal length of the camera is {focal_length_mm} mm")
                
                # h,  w = img.shape[:2]
                # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
                # dst = cv2.undistort(img, cameraMatrix, dist, None, newcameramtx)
                # # crop the image
                # x, y, w, h = roi
                # dst = dst[y:y+h, x:x+w]
                # cv2.imwrite('calibresult.png', dst)
                # ............................................................................................

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                # print("found connection")
                # cx, cy = sort_leds(cx, cy)
                # print("sorted")
                # calulate_angle(cx, cy)
                # except Exception:
                # print("exception thrown")
                # pass
                # else: 
                # cv2.line(img, (cx[0], cy[0]), (cx[1], cy[1]), (0, 255, 0), 4 )
                # # cv2.line(img, (cx[0], cy[0]), (cx[2], cy[2]), (0, 255, 0), 2 )
                # cv2.line(img, (cx[0], cy[0]), (cx[3], cy[3]), (0, 255, 0), 4 )
                # cv2.line(img, (cx[0], cy[0]), (cx[4], cy[4]), (0, 255, 0), 4 )
                # cv2.line(img, (cx[1], cy[1]), (cx[2], cy[2]), (0, 255, 0), 4 )
                # # cv2.line(img, (cx[1], cy[1]), (cx[4], cy[4]), (0, 255, 0), 2 )
                # cv2.line(img, (cx[2], cy[2]), (cx[4], cy[4]), (0, 255, 0), 4 )
                # cv2.line(img, (cx[2], cy[2]), (cx[3], cy[3]), (255, 0, 0), 4 )
                # cv2.line(img, (cx[1], cy[1]), (cx[3], cy[3]), (0, 0, 255), 2 )
                # cv2.line(img, (cx[3], cy[3]), (cx[4], cy[4]), (0, 0, 255), 2 )

            
        
        # # Press Q on keyboard to exit
        if  cv2.waitKey(0) and 0xFF == ord('q'):
            break

    else:
        print("Fin")
        break

    

# When everything done, release
# the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

#Statistics
print("Total frames:      ", frames_count)
print("Frames per Second: ", fps)
print("Duration:          ", time, " seconds")