import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(config)
picam2.start()
time.sleep(2)
while (True):
    frame = picam2.capture_array()
    # frame = frame[1920*0.2 : 1920*0.8 , 1080*0.2:1080*0.2]
    cv2.imshow("Display", frame)
    time.sleep(2)