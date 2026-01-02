import cv2
from picamera2 import Picamera2
import time

cam = Picamera2()

cam.preview_configuration.main.size = (1280, 720)
cam.preview_configuration.main.format = "RGB888"
cam.configure("preview")
cam.start()
time.sleep(2)
meta = cam.capture_metadata()
cam.set_controls({
    "FrameDurationLimits": (500_000, 500_000),
    "ExposureTime": 50000,
    "AnalogueGain": 1.0,
    "ColourGains": meta["ColourGains"],
    "AeEnable": False,
    "AwbEnable": False,
})
time.sleep(1)

frameNum = 0
while (True):
    if input("continue(takeimage)" + str(frameNum) + ":(y/n)") == "n":
        break
    frame = cam.capture_array()
    frame = frame[404:585, 2:1078]
    cv2.imwrite("Display"+str(frameNum)+".png", frame)
    cv2.imshow("Display", frame)
    # time.sleep(2)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    frameNum += 1
    # exposureTime = input("exposureTime:")
    # print("currentExposureTime", exposureTime)
cam.stop()
cv2.destroyAllWindows()