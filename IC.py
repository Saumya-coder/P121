import cv2
import numpy as np
import time 

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)
time.sleep(10)
bg = 0

for i in range(60):
    ret,bg = cap.read()
bg =  np.flip(bg, axis = 1)


while (cap.isOpened()):
    ret, image = cap.read()
    if not ret :
        break
    image = np.flip(image, axis = 1)

    # how to detect black color and how to make it invisible
    # Convert the way in which colors are seen
    
    hsv =   cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)
    mask_1 = cv2.inRange(hsv, lower_black, upper_black)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    mask_2 = cv2.bitwise_not(mask_1)
    res_1 = cv2.bitwise_and(image,image, mask = mask_2)
    res_2 = cv2.bitwise_and(bg,bg, mask = mask_1)
    finalOutput = cv2.addWeighted(res_1,1, res_2,1, 0)
    output_file.write(finalOutput)
    cv2.imshow('Magic', finalOutput)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()    


