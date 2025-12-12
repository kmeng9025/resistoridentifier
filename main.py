import cv2
from picamera2 import Picamera2
import time

cam = Picamera2()
cam.preview_configuration.main.size = (1280, 720)
cam.preview_configuration.main.format = "RGB888"
cam.configure("preview")
cam.start()
time.sleep(2)
while (True):
    frame = cam.capture_array()
    frame = frame[364:545, 44 : 1120]
    pre_bil = cv2.bilateralFilter(frame, 5, 80, 80)
    hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 59, 5)
    thresh = cv2.bitwise_not(thresh)
    cv2.imshow("Display", thresh)
    time.sleep(2)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cam.stop()
cv2.destroyAllWindows()