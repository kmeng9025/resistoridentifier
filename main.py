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
    # frame = frame[1920*0.2 : 1920*0.8 , 1080*0.2:1080*0.2]
    cv2.imshow("Display", frame)
    time.sleep(2)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()