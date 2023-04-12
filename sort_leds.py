import numpy as np
import scipy
import math

def collinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-12


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
    mid_points =[] #points that are in the middle of the bounding box
    for _ in range(0, 5):
        if _ not in min_max_coord:
            mid_points.append(_)
            
  
    #if the middle point is known (unlikely)
    if len(mid_points) == 1:
        print(" sort_leds error: one mid led was found")
        return [None] * 5, [None] * 5
        

    elif len(mid_points) == 2 :        
        linex = [cx[mid_points[0]], cx[mid_points[1]]]  #two cx points of mid points
        liney = [cy[mid_points[0]], cy[mid_points[1]]]  #two cy points of mid points
        linex.append(None)
        liney.append(None)
        max_r = -2 #value is between -1 and +1
        max_score = 0
        for _ in range(0, 5):
            print("test 1")
            score = sum(collinear((cx[mid_points[0]], cy[mid_points[0]]),
                                    (cx[mid_points[1]], cy[mid_points[1]]),
                                      (cx[_], cy[_])))
            print("test 2")
            #found botttom led
            if score >= max_score:
                max_Score = score
                sorted_cx[4] = cx[_]
                sorted_cy[4] = cy[_]



            # if _ in min_max_coord:
            #     linex[2] = cx[_]
            #     liney[2] = cy[_]
            #     pearsons_r = scipy.stats.pearsonr(linex, liney)[0]
            #     #new max
            #     if pearsons_r > max_r:
            #         max_r = pearsons_r
            #         #found botttom led
            #         sorted_cx[4] = cx[_]
            #         sorted_cy[4] = cy[_]

        #found #4 now find others 
        distance0 = math.dist( (sorted_cx[4], sorted_cy[4]) , (cx[mid_points[0]], cy[mid_points[0]]  ))
        distance1 = math.dist( (sorted_cx[4], sorted_cy[4]) , (cx[mid_points[1]], cy[mid_points[1]]  ))   
        if distance0 < distance1:
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
        #Now to find led 0 and 2
        for _ in range(0,5):
            if cx[_] not in sorted_cx and cy[_] not in sorted_cy:

                if sorted_cx[4] == sorted_cx[1]:
                    print("Error coordinates are incorrect")
                    return [None] * 5, [None] * 5

                elif sorted_cx[4] < sorted_cx[1]:
                    
                    if sorted_cy[1] < cy[_]:
                        sorted_cx[2] = cx[_]
                        sorted_cy[2] = cy[_]
                    elif sorted_cy[1] > cy[_]:
                        sorted_cx[0] = cx[_]
                        sorted_cy[0] = cy[_]
                    else:
                        print("mot sure if this is possible")
                        if sorted_cx[1] > cx[_]:
                            sorted_cx[0] = cx[_]
                            sorted_cy[0] = cy[_]
                        else:
                            sorted_cx[2] = cx[_]
                            sorted_cy[2] = cy[_]
                else:
                    
                   
                    if sorted_cy[1] < cy[_]:
                        sorted_cx[0] = cx[_]
                        sorted_cy[0] = cy[_]
                    else:
                        sorted_cx[2] = cx[_]
                        sorted_cy[2] = cy[_]

    #TODO
    elif len(mid_points) == 3 :
        print("3 leds in the middle")
        return [None] * 5, [None] * 5
    else:
        print("Error finding points\n")  
        return [None] * 5, [None] * 5

    return sorted_cx, sorted_cy
