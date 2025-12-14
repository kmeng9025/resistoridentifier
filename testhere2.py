import cv2
import numpy as np

# ---------- Resistor decoding ----------
DIGIT = {
    "black":0,"brown":1,"red":2,"orange":3,"yellow":4,
    "green":5,"blue":6,"violet":7,"gray":8,"white":9
}
MULT = {
    "silver":1e-2,"gold":1e-1,"black":1,"brown":10,"red":100,"orange":1e3,"yellow":1e4,
    "green":1e5,"blue":1e6,"violet":1e7,"gray":1e8,"white":1e9
}
TOL = {
    "brown":1,"red":2,"green":0.5,"blue":0.25,"violet":0.1,"gray":0.05,
    "gold":5,"silver":10
}

def decode_resistor(bands):
    """
    bands: list like ["brown","black","red","gold"] (4-band)
           or ["brown","black","black","red","brown"] (5-band)
    returns (ohms, tolerance_percent)
    """
    b = [x.lower() for x in bands]
    if len(b) == 4:
        d1, d2, mult, tol = b
        value = (10*DIGIT[d1] + DIGIT[d2]) * MULT[mult]
        return value, TOL.get(tol, None)
    elif len(b) == 5:
        d1, d2, d3, mult, tol = b
        value = (100*DIGIT[d1] + 10*DIGIT[d2] + DIGIT[d3]) * MULT[mult]
        return value, TOL.get(tol, None)
    else:
        raise ValueError("Expected 4 or 5 bands")

def format_ohms(v):
    if v >= 1e6: return f"{v/1e6:.3g} MΩ"
    if v >= 1e3: return f"{v/1e3:.3g} kΩ"
    return f"{v:.3g} Ω"

# ---------- Rectangle trimming ----------
def rectified_vertical_rectangles(binary_mask,
                                 min_area=300,
                                 min_aspect=2.0,          # height/width
                                 min_rectangularity=0.75, # contour_area / rect_area
                                 max_tilt_deg=20):
    """
    Returns a mask where each blob is replaced by its best-fit vertical rectangle.
    This trims 'rectangle + attached blob' into just a rectangle.
    """
    m = (binary_mask > 0).astype(np.uint8) * 255
    out = np.zeros_like(m)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if w < 1 or h < 1:
            continue

        # normalize so h is long side
        if w > h:
            w, h = h, w
            ang += 90

        aspect = h / w
        if aspect < min_aspect:
            continue

        rect_area = w * h
        rectangularity = area / rect_area
        if rectangularity < min_rectangularity:
            continue

        # keep near-vertical rectangles
        # treat 90deg as vertical; accept within max_tilt_deg
        # normalize angle to [0, 180)
        a = ang % 180
        tilt = min(abs(a - 90), abs(a - 270))
        if tilt > max_tilt_deg:
            continue

        box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
        cv2.fillPoly(out, [box], 255)

    return out

frame = cv2.imread("test.png")

# Faster than heavy bilateral on Pi Zero
blur = cv2.GaussianBlur(frame, (5, 5), 0)

hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

lower_blue = np.array([80, 120, 50])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Your code inverts blue; keep consistent:
inv = cv2.bitwise_not(blue_mask)

# Clean mask a bit
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)

# Fast crop bounding box (replaces your nested loops) :contentReference[oaicite:4]{index=4}
ys, xs = np.where(inv == 0)  # you were cropping around zeros
if len(ys) > 0:
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    frame_c = frame[y0:y1, x0:x1]
    inv_c = inv[y0:y1, x0:x1]
else:
    frame_c, inv_c = frame, inv

# Replace each blob with a vertical rectangle (trims off glint bumps)
rect_mask = rectified_vertical_rectangles(inv_c,
                                         min_area=300,
                                         min_aspect=2.0,
                                         min_rectangularity=0.75,
                                         max_tilt_deg=20)

# OPTIONAL: extract each rectangle ROI and compute average color per band
cnts, _ = cv2.findContours(rect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = []
for c in cnts:
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    boxes.append((cx, cy, w, h, ang))

# Sort left-to-right (typical band order)
boxes.sort(key=lambda t: t[0])

band_bgr = []
for (cx, cy, w, h, ang) in boxes:
    box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
    m = np.zeros(rect_mask.shape, np.uint8)
    cv2.fillPoly(m, [box], 255)
    pixels = frame_c[m == 255]
    if len(pixels) == 0:
        continue
    bgr = np.median(pixels, axis=0)  # median is robust to glare
    band_bgr.append(tuple(map(float, bgr)))

# TODO: map band_bgr -> band color names using your classifier
# For now, if you already have them:
bands = ["brown", "black", "red", "gold"]  # <-- replace with your detected band names
ohms, tol = decode_resistor(bands)
print("Bands:", bands)
print("Value:", format_ohms(ohms), (f"±{tol}%" if tol is not None else ""))

# Visualize
result = cv2.bitwise_and(frame_c, frame_c, mask=rect_mask)
cv2.imshow("Rect-mask", rect_mask)
cv2.imshow("Rectified result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
