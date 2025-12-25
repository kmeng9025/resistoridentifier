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

exposureTime = 50000
while (True):
    frame = cam.capture_array()
    frame = frame[364:545, 44 : 1120]
    cv2.imshow("Display", frame)
    time.sleep(2)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    exposureTime = input("exposureTime:")
    cam.set_controls({
        "FrameDurationLimits": (1_000_000, 1_000_000),
        "ExposureTime": int(exposureTime),
        "AnalogueGain": 1.0,
        "ColourGains": meta["ColourGains"],
        "AeEnable": False,
        "AwbEnable": False,
    })
    time.sleep(1.1)
    print("currentExposureTime", exposureTime)
cam.stop()
cv2.destroyAllWindows()