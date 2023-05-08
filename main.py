import itertools
from matplotlib import pyplot as plt
import numpy as np
import cv2

from sort_leds import sort_leds
# from calculate_angle import calculate_angle
from process_img import find_connections
from rotate import rotate_image


VideoPath_1 = r"D:\John\Project\Trial1\Cal 02.2144931.20220906113403.avi"
# VideoPath_1 = r"D:\John\Project\GOPRO\Cal02214490420220906113403.avi"
template = cv2.imread('D:\John\Project\Code\masters-project\wtemplate.png') 
START_FRAME = 500

#       0   1   2
#           3      
#           4
#          
# Geometric constraints
mm_wand_length = {
    (0, 1): 160,
    (0, 2): 240,
    (0, 3): 203,
    (0, 4): 293,
    (1, 2): 80,
    (1, 3): 125,
    (1, 4): 245,
    (2, 3): 148,
    (2, 4): 258,
    (3, 4): 120
}

# Temporal consistency
persistent_leds = []

def draw_box(event, x, y, flags, param):
    global drawing, top_left, bottom_right

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            bottom_right = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right = (x, y)
        
        
drawing = False
top_left = (0, 0)
bottom_right = (0, 0)

def filter_contours_by_box(contours, top_left, bottom_right):
    filtered_contours = []

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if top_left[0] < cX < bottom_right[0] and top_left[1] < cY < bottom_right[1]:
                filtered_contours.append(cnt)
    
    return filtered_contours

def preprocess_image(img):
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    fgmask = fgbg.apply(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    # blurred = cv2.GaussianBlur(img, (5, 5), 0)  # add Gaussian blur

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
 
    # blurred = cv2.medianBlur(img, 3) 
    edges = cv2.Canny(img, 100, 200)

    return edges

def detect_leds(img, fgmask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 240, 120])
    upper_red1 = np.array([5, 255, 255])

    lower_red2 = np.array([175, 240, 120])
    upper_red2 = np.array([180, 255, 255])

    lower_white = np.array([0, 0, 240])
    upper_white = np.array([255, 200, 255])


    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_white)
    mask = cv2.bitwise_and(mask, fgmask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    led_contours = []
    excluded_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / perimeter**2 if perimeter != 0 else 0

        if  50 < area < 500 and circularity > 0.5 :#and circularity < 0.25:
            led_contours.append(contour)
        else:
            excluded_contours.append(contour)

    return led_contours, excluded_contours

def filter_leds_by_geometric_constraints(leds):
    filtered_leds = []
    for (i1, led1), (i2, led2) in itertools.combinations(enumerate(leds), 2):
        M1 = cv2.moments(led1)
        M2 = cv2.moments(led2)
        cX1 = int(M1["m10"] / M1["m00"])
        cY1 = int(M1["m01"] / M1["m00"])
        cX2 = int(M2["m10"] / M2["m00"])
        cY2 = int(M2["m01"] / M2["m00"])

        distance = np.sqrt((cX2 - cX1)**2 + (cY2 - cY1)**2)
        # print(distance)

        pair_index = (i1, i2)
        # print(pair_index)

        if pair_index in mm_wand_length:
            # if mm_wand_length[pair_index] * 0.2 >= distance :#<= mm_wand_length[pair_index] * 1.75:
             if  distance >= mm_wand_length[pair_index] * 1.75:
                filtered_leds.extend([led1, led2])
    return filtered_leds

def update_persistent_leds(leds, persistent_leds, persistence_threshold=2):
    current_leds = [cv2.boundingRect(led) for led in leds]
    updated_persistent_leds = []

    for led in current_leds:
        found_match = False
        for i, (p, count) in enumerate(persistent_leds):
            if overlap(led, p):
                updated_persistent_leds.append((led, count + 1))
                found_match = True
                break
        if not found_match:
            updated_persistent_leds.append((led, 1))

    persistent_leds = [p for p in updated_persistent_leds if p[1] >= persistence_threshold]
    return persistent_leds

def overlap(rect1, rect2):
    # print("rect1: ", rect1)
    # print("rect2: ", rect2)
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return (x1 < x2 + w2) and (x2 < x1 + w1) and (y1 < y2 + h2) and (y2 < y1 + h1)


# def main():
# print("tests")
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
while (cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
       
       
        frame_progress += 1
        if frame_progress == START_FRAME:
            print("ready")
            # cv2.waitKey(0)
        if frame_progress > START_FRAME:
            fgmask = fgbg.apply(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            img_height, img_width = img.shape[:2]
            

            detected_leds, excludes_leds = detect_leds(img, fgmask)
            cv2.drawContours(img, detected_leds, -1, (255, 0, 0), -1)
            cv2.drawContours(img, excludes_leds, -1, (0, 0, 255), -1)
            cv2.imshow('image', img)
            cv2.setMouseCallback('image', draw_box)
            img_copy = img.copy()
            cx = []
            cy = []
            while True:
                
                if drawing:
                    img_copy = img.copy()
                    cv2.rectangle(img_copy, top_left, bottom_right, (255, 0, 0), 2)
                    box_contours = filter_contours_by_box(detected_leds,top_left, bottom_right)
                    cv2.drawContours(img_copy, box_contours, -1, (0, 255, 0), -1)
                    
                cv2.imshow('image', img_copy)

                key = cv2.waitKey(1)
                if key == 27:  # Esc key
                    break
                elif key == 121:
                    if len(box_contours) ==5:  # Press 'y' to save the selection
                        print('Selection confirmed.')
                        cx = []
                        cy = []
                        for cnt in box_contours:
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cx.append( int(M["m10"] / M["m00"]))
                                cy.append(int(M["m01"] / M["m00"]))     
                        print(cx)
                        print(cy)
                        break
                    else:
                        print("Wrong number of leds")
                elif key == 110: #press 'n'
                    print('Redo the selection.')
            

            blank_filter= np.zeros((img_height, img_width, 3), dtype=np.uint8)
            filtered_leds = filter_leds_by_geometric_constraints(detected_leds)
            cv2.drawContours(blank_filter,  filtered_leds, -1, (255, 255, 255), -1)
            # cv2.imshow('filter', blank_filter)

            blank_persistent= np.zeros((img_height, img_width, 3), dtype=np.uint8)
            persistent_leds = update_persistent_leds(detected_leds, persistent_leds)        
            cv2.drawContours(blank_persistent, persistent_leds, -1, (255, 0, 0), -1)
            # cv2.imshow('persisntent', blank_persistent)
            # match_leds(template,img )


            if cx and cy:
                cx, cy = sort_leds(cx, cy)
                print(cx)
                print(cy)
                cv2.line(img_copy, (cx[0], cy[0]), (cx[1], cy[1]), (255, 0, 0), 4 )
                cv2.line(img_copy, (cx[1], cy[1]), (cx[2], cy[2]), (255, 0, 0), 4 )
                cv2.line(img_copy, (cx[1], cy[1]), (cx[3], cy[3]), (0, 0, 255), 2 )
                cv2.line(img_copy, (cx[3], cy[3]), (cx[4], cy[4]), (0, 0, 255), 2 )

                # # cv2.line(img, (cx[0], cy[0]), (cx[2], cy[2]), (0, 255, 0), 2 )
                # cv2.line(img_copy, (cx[0], cy[0]), (cx[2], cy[2]), (0, 255, 0), 4 )
                # cv2.line(img_copy, (cx[0], cy[0]), (cx[4], cy[4]), (0, 255, 0), 4 )
                
                # # cv2.line(img, (cx[1], cy[1]), (cx[4], cy[4]), (0, 255, 0), 2 )
                # cv2.line(img_copy, (cx[2], cy[2]), (cx[4], cy[4]), (0, 255, 0), 4 )
                # cv2.line(img_copy, (cx[2], cy[2]), (cx[3], cy[3]), (255, 0, 0), 4 )
                
                
                cv2.imshow('image', img_copy)
                k = cv2.waitKey(0)

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
            # img_points = np.array([[495, 474], [436, 443], [412, 426], [461, 396], [482, 353]], dtype=np.float32)
            # real_points1 = np.float32([[0, 0,0 ], [160, 0,0], [240, 0,0], [160, 80,0], [160, 245,0]])
            # img_points = []
            # obj_points = []
            # camera_matrix = np.zeros((3, 3))
            # dist_coeffs = np.zeros((5, 1))

            # image_points = np.array([[495, 474], [436, 443], [412, 426], [461, 396], [482, 353]], dtype=np.float32)
            # img_points.append(image_points)  
            # image_points = np.array([[492, 323], [468, 323], [450, 282], [489, 301], [517, 282]], dtype=np.float32)
            # img_points.append(image_points) 
            # image_points = np.array([[776, 481], [706, 477], [670, 474], [711, 418], [715, 362]], dtype=np.float32)
            # img_points.append(image_points) 
            # image_points = np.array([[1066, 12], [1066, 53], [1066, 70], [1031, 78], [993, 105]], dtype=np.float32)
            # img_points.append(image_points) 
            # image_points = np.array([[1045, 218], [1017, 270], [1003, 297], [987, 262], [955, 252]], dtype=np.float32)
            # img_points.append(image_points) 
            # image_points = np.array([[305, 532], [277, 482], [263, 455], [308, 436], [334, 390]], dtype=np.float32)
            # img_points.append(image_points) 
            # image_points = np.array([[235, 436], [195, 367], [175, 335], [251, 332], [311, 291]], dtype=np.float32)
            # img_points.append(image_points)  
            # image_points = np.array([[1021, 127], [1007, 183], [1001, 215], [964, 188], [917, 196]], dtype=np.float32)
            # img_points.append(image_points)  
            # image_points = np.array([[1000, 336], [962, 3773], [936, 402], [920, 354], [877, 335]], dtype=np.float32)
            # img_points.append(image_points) 
            # image_points = np.array([[562, 641], [621, 652], [656, 655], [605, 680], [586, 713]], dtype=np.float32)
            # img_points.append(image_points) 
            # object_points = np.array([[0, 0,0 ], [160, 0,0], [240, 0,0], [160, 80,0], [160, 245,0]], dtype=np.float32)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            # obj_points.append(object_points)
            
            
            # retval, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (img_height, img_width), None, None)      
            # focal_length_px = cameraMatrix[0, 0]
            # print(f"The focal length of the camera is {focal_length_px} pixels")

            # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
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



        if cv2.waitKey(1) and 0xFF == ord('q'):
            break


    else:
        print("Fin")
        break
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

#Statistics
print("Total frames:      ", frames_count)
print("Frames per Second: ", fps)
print("Duration:          ", time, " seconds")