import math

def calulate_angle(cx, cy):

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
        print("upside down")
    else:
        print("right way")

    if cx[3] < cx[1]: 
        print("facing right")
    else:
        print("facing left")

    diff_cx = cx[1] - cx[3]
    diff_cy = cy[1] - cy[3]   
    handle_angle1 = (diff_cy/diff_cx)

    diff_cx = cx[3] - cx[4]
    diff_cy = cy[3] - cy[4]   
    handle_angle2 = (diff_cy/diff_cx)

    diff_cx = cx[1] - cx[4]
    diff_cy = cy[1] - cy[4]   
    handle_angle3 = (diff_cy/diff_cx)

   #TODO FIND ERROR
    
 
    print("handle: ", handle_angle1)
    print("handle: ", handle_angle2)
    print("handle: ", handle_angle3)
 
    # print("avg     ", avg_angle)
    print("......................")

    print ("0 TO 1: ", pixel_wand_lenth_0_to_1 / mm_wand_lenth_0_to_1)
    print ("0 TO 2: ", pixel_wand_lenth_0_to_2 / mm_wand_lenth_0_to_2)
    print ("0 TO 3: ", pixel_wand_lenth_0_to_3 / mm_wand_lenth_0_to_3)
    print ("0 TO 4: ", pixel_wand_lenth_0_to_4 / mm_wand_lenth_0_to_4)
    print ("1 TO 2: ", pixel_wand_lenth_1_to_2 / mm_wand_lenth_1_to_2)
    print ("1 TO 3: ", pixel_wand_lenth_1_to_3 / mm_wand_lenth_1_to_3)
    print ("1 TO 4: ", pixel_wand_lenth_1_to_4 / mm_wand_lenth_1_to_4)
    print ("2 TO 3: ", pixel_wand_lenth_2_to_3 / mm_wand_lenth_2_to_3)
    print ("2 TO 4: ", pixel_wand_lenth_2_to_4 / mm_wand_lenth_2_to_4)
    print ("3 TO 4: ", pixel_wand_lenth_3_to_4 / mm_wand_lenth_3_to_4)

    # print( mm_wand_lenth_3_to_4 / mm_wand_lenth_1_to_3)

    
    return 
