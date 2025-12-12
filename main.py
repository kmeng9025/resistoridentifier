# import cv2
# from picamera2 import Picamera2
# import time

# cam = Picamera2()
# cam.preview_configuration.main.size = (1280, 720)
# cam.preview_configuration.main.format = "RGB888"
# cam.configure("preview")
# cam.start()
# time.sleep(2)
# while (True):
#     frame = cam.capture_array()
#     # frame = frame[1920*0.2 : 1920*0.8 , 1080*0.2:1080*0.2]
#     cv2.imshow("Display", frame)
#     time.sleep(2)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cam.stop()
# cv2.destroyAllWindows()
import cv2
import numpy as np
from picamera2 import Picamera2
import time

# ---------- Color helpers ----------

# Rough mapping color name -> digit/tolerance
RESISTOR_DIGITS = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "gray": 8,
    "white": 9,
}

RESISTOR_MULTIPLIERS = {
    "black": 1,
    "brown": 10,
    "red": 100,
    "orange": 1_000,
    "yellow": 10_000,
    "green": 100_000,
    "blue": 1_000_000,
    "violet": 10_000_000,
    "gray": 100_000_000,
    "white": 1_000_000_000,
    "gold": 0.1,
    "silver": 0.01,
}

RESISTOR_TOLERANCE = {
    "brown": 1,
    "red": 2,
    "green": 0.5,
    "blue": 0.25,
    "violet": 0.1,
    "gray": 0.05,
    "gold": 5,
    "silver": 10,
}


def classify_color(hsv):
    """Very rough HSV -> resistor color name classifier.
       You WILL want to tweak thresholds for your lighting.
    """
    h, s, v = hsv

    # black / white / gray first
    if v < 40:
        return "black"
    if s < 40 and v > 200:
        return "white"
    if s < 40:
        return "gray"

    # Brown is basically dark orange
    if 8 < h < 30 and v < 120:
        return "brown"

    # Gold / silver (low saturation, mid value) – rarely needed
    if s < 80 and 120 < v < 200:
        # more yellow-ish
        if 15 < h < 45:
            return "gold"
        else:
            return "silver"

    # Hue-based colors (0-179 in OpenCV)
    if h < 8 or h > 170:
        return "red"
    if 8 <= h < 25:
        return "orange"
    if 25 <= h < 35:
        return "yellow"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 130:
        return "blue"
    if 130 <= h < 160:
        return "violet"

    return "unknown"


def decode_resistor(colors):
    """
    Interpret a list of color names as a 4- or 5-band resistor.
    Try both directions (because we may not know which end is tolerance).
    Returns a list of possible (value_ohms, tolerance_percent, direction).
    """
    results = []

    def decode_one(seq, direction):
        # strip "unknown"
        seq = [c for c in seq if c != "unknown"]
        if len(seq) < 4:
            return

        # assume 4-band: [d1, d2, multiplier, tolerance]
        c = seq[:4]
        if all(col in RESISTOR_DIGITS for col in c[:2]) and c[2] in RESISTOR_MULTIPLIERS:
            d1 = RESISTOR_DIGITS[c[0]]
            d2 = RESISTOR_DIGITS[c[1]]
            mult = RESISTOR_MULTIPLIERS[c[2]]
            value = (10 * d1 + d2) * mult
            tol = RESISTOR_TOLERANCE.get(c[3], None)
            results.append((value, tol, direction))

        # assume 5-band: [d1, d2, d3, multiplier, tolerance]
        if len(seq) >= 5:
            c = seq[:5]
            if all(col in RESISTOR_DIGITS for col in c[:3]) and c[3] in RESISTOR_MULTIPLIERS:
                d1 = RESISTOR_DIGITS[c[0]]
                d2 = RESISTOR_DIGITS[c[1]]
                d3 = RESISTOR_DIGITS[c[2]]
                mult = RESISTOR_MULTIPLIERS[c[3]]
                value = (100 * d1 + 10 * d2 + d3) * mult
                tol = RESISTOR_TOLERANCE.get(c[4], None)
                results.append((value, tol, direction))

    decode_one(colors, "L→R")
    decode_one(list(reversed(colors)), "R→L")
    return results


# ---------- Band detection from a frame ----------

def find_bands(frame):
    """
    Given a BGR frame, find color bands along the resistor.
    Returns: list of (x_start, x_end, color_name)
    """

    h, w, _ = frame.shape

    # Crop a horizontal strip where the resistor is.
    # You may need to tune these values depending on slit position.
    y1 = int(h * 0.55)
    y2 = int(h * 0.75)
    strip = frame[y1:y2, :]

    # Smooth and convert to HSV
    blur = cv2.GaussianBlur(strip, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Average each column over Y -> get 1D color profile (w x 3)
    col_mean = hsv.mean(axis=0)

    # Look at color differences between neighbouring columns
    diff = np.linalg.norm(np.diff(col_mean, axis=0), axis=1)

    # Threshold for "big change" – tune this!
    # using mean + some factor of std is a bit more adaptive
    thr = diff.mean() + 1.5 * diff.std()

    segments = []
    start = 0
    min_width = 5  # ignore very tiny segments (noise)

    for x in range(1, w):
        if diff[x - 1] > thr:
            if x - start >= min_width:
                segments.append((start, x))
            start = x
    if w - start >= min_width:
        segments.append((start, w))

    # Compute avg color for each segment and classify
    bands = []
    for (xs, xe) in segments:
        seg_hsv = col_mean[xs:xe].mean(axis=0)
        color = classify_color(seg_hsv)
        width = xe - xs

        # Heuristic: ignore very wide segments (likely body or background)
        # and very narrow ones (noise). You can adjust these.
        if width < 15 or width > 250:
            continue

        bands.append((xs, xe, color))

    # Optional: merge consecutive segments of same color
    merged = []
    for xs, xe, color in bands:
        if merged and merged[-1][2] == color and xs - merged[-1][1] < 10:
            # extend last segment
            last_xs, last_xe, last_color = merged[-1]
            merged[-1] = (last_xs, xe, last_color)
        else:
            merged.append((xs, xe, color))

    return merged


# ---------- Main camera loop ----------

def main():
    cam = Picamera2()
    cfg = cam.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(2)

    print("Press 'q' to quit, 's' to force a reading.")
    last_print = 0

    while True:
        frame = cam.capture_array()
        display = frame.copy()

        # Find bands every ~0.5 s
        now = time.time()
        do_read = (now - last_print > 0.5)

        bands = find_bands(frame)

        # Draw bands on the display frame
        for xs, xe, color in bands:
            cv2.rectangle(display, (xs, 0), (xe, display.shape[0]), (0, 0, 0), 2)
            cv2.putText(display, color, (xs + 2, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if do_read and bands:
            last_print = now
            color_seq = [c for _, _, c in bands]
            print("Detected band colors:", color_seq)
            guesses = decode_resistor(color_seq)

            if guesses:
                for value, tol, direction in guesses:
                    if value >= 1_000_000:
                        txt = f"{value/1_000_000:.2f} MΩ"
                    elif value >= 1_000:
                        txt = f"{value/1_000:.2f} kΩ"
                    else:
                        txt = f"{value:.0f} Ω"
                    if tol is not None:
                        txt += f" ±{tol}%"
                    print(f"  {direction}: {txt}")
            else:
                print("  Could not decode a valid resistor from colors.")

        cv2.imshow("Resistor", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            last_print = 0  # force immediate read

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
