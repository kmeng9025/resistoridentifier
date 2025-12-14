import cv2
import time
import numpy as np
import cv2
import numpy as np

frame = cv2.imread("test.png")
pre_bil = cv2.bilateralFilter(frame, 10, 10, 200)
hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
lower_blue = np.array([80, 120, 50])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
inverse_mask = cv2.bitwise_not(blue_mask)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
inverse_mask = cv2.dilate(inverse_mask, kernel, iterations=3)
inverse_mask = cv2.erode(inverse_mask, kernel, iterations=5)
cropped = [-1, -1, -1, -1]
last = [0, 0]
file = open("out.txt", "w")
for i in range(len(inverse_mask)):
    for j in range(len(inverse_mask[i])):
        if(cropped[0] == -1 and inverse_mask[i][j] == 0):
            cropped[0] = i
        if(cropped[1] == -1 and inverse_mask[i][j] == 0):
            cropped[1] = j
        if(inverse_mask[i][j] == 255):
            file.write("1")
        else:
            file.write(str(inverse_mask[i][j])) 
            last = [i, j] 
    file.write("\n")
file.close()
cropped[2:] = last
pre_bil_cropped = pre_bil[cropped[0] : cropped[2], cropped[1] : cropped[3]]
inverse_mask = inverse_mask[cropped[0] : cropped[2], cropped[1] : cropped[3]]
oned = False
rectangles = []
file = open("out2.txt", "w")
for j in range(len(inverse_mask[0])):
    if(inverse_mask[0][j] == 0 and oned):
        rectangles[-1].append(0)
        rectangles[-1].append(j)
        oned = False
    elif(inverse_mask[0][j] == 255 and not oned):
        oned = True
        rectangles.append([0, j])
        file.write(str(inverse_mask[0][j]))
file.close()
print(inverse_mask[0][0])
print(rectangles)
averages = []
themask = np.zeros((inverse_mask.__len__(), inverse_mask[0].__len__()), np.uint8)
try:
    for i in rectangles:
        themask [i[0] :len(inverse_mask), (i[1] + int(abs(i[3]-i[1])/2.5)):(i[3] - int(abs(i[3]-i[1])/2.5))] = 1
        print(np.mean(np.mean(pre_bil_cropped[i[0] :len(inverse_mask), (i[1] + int(abs(i[3]-i[1])/2.5)):(i[3] - int(abs(i[3]-i[1])/2.5))], 0), 0))
except Exception as e:
    pass
result = cv2.bitwise_and(pre_bil_cropped, pre_bil_cropped, mask=themask)
cv2.imshow("Display2", frame)
cv2.imshow("Display", result)
cv2.imshow("Display3", pre_bil_cropped)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()