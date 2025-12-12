import cv2
import numpy as np
from picamera2 import Picamera2
import time


# ------------------  COLOR TABLE --------------------
COLOR_TABLE = {
    "black":  (0, 0,  20),
    "brown":  (10, 180, 80),
    "red":    (0, 220, 200),
    "orange": (12, 220, 230),
    "yellow": (26, 220, 230),
    "green":  (60, 220, 200),
    "blue":   (110, 220, 200),
    "violet": (140, 220, 180),
    "gray":   (0, 20, 150),
    "white":  (0, 10, 240),
    "gold":   (22, 80, 180),
    "silver": (0, 10, 190),
}

def classify(avg_hsv):
    hsv = np.array(avg_hsv)
    best = None
    bestdist = 99999
    for name, ref in COLOR_TABLE.items():
        d = np.linalg.norm(hsv - np.array(ref))
        if d < bestdist:
            bestdist = d
            best = name
    return best


# ------------------  MAIN PIPELINE --------------------
def detect_resistor_bands(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape

    # 1. Find row with highest saturation variance → resistor row
    sat = hsv[:, :, 1]
    var = sat.var(axis=1)
    row = int(np.argmax(var))

    # 2. Extract strip around resistor body
    strip = hsv[row-6:row+6, :, :]   # just 12 px tall
    profile = strip.mean(axis=0)     # shape (w, 3)

    # 3. Find resistor body region = longest region of high saturation
    sat_line = profile[:,1]
    mask = sat_line > 80             # adjust threshold if needed

    # find longest contiguous True region
    best_start = best_end = 0
    cur_start = None
    for i, v in enumerate(mask):
        if v and cur_start is None:
            cur_start = i
        if not v and cur_start is not None:
            if i - cur_start > best_end - best_start:
                best_start, best_end = cur_start, i
            cur_start = None
    # close final segment
    if cur_start is not None:
        if w - cur_start > best_end - best_start:
            best_start, best_end = cur_start, w

    body_hsv = profile[best_start:best_end]
    body_color = body_hsv.mean(axis=0)

    # 4. Compute how far each pixel's hue deviates from body hue
    hue = profile[:,0]
    base_h = body_color[0]
    diff = np.abs(hue - base_h)
    diff = cv2.GaussianBlur(diff.reshape(1,-1), (1,31), 0)[0]

    # 5. Peaks in deviation = bands
    thresh = diff.mean() + diff.std()
    band_positions = np.where(diff > thresh)[0]

    # group into segments
    bands = []
    if len(band_positions) > 0:
        start = band_positions[0]
        for i in range(1, len(band_positions)):
            if band_positions[i] != band_positions[i-1] + 1:
                bands.append((start, band_positions[i-1]))
                start = band_positions[i]
        bands.append((start, band_positions[-1]))

    # 6. classify bands by average HSV
    colors = []
    for xs, xe in bands:
        seg_hsv = profile[xs:xe].mean(axis=0)
        cname = classify(seg_hsv)
        colors.append((xs, xe, cname))

    return colors, body_color, row, best_start, best_end


# ------------------ CAMERA LOOP --------------------
cam = Picamera2()
cfg = cam.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
cam.configure(cfg)
cam.start()
time.sleep(2)

while True:
    frame = cam.capture_array()
    colors, body_hsv, row, s, e = detect_resistor_bands(frame)
    disp = frame.copy()

    # draw resistor body region
    cv2.rectangle(disp, (s, row-6), (e, row+6), (0,255,0), 2)

    # draw bands
    for xs, xe, cname in colors:
        cv2.rectangle(disp, (xs, row-10), (xe, row+10), (0,0,0), 2)
        cv2.putText(disp, cname, (xs, row-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imshow("Resistor", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
