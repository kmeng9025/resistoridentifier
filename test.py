import cv2
import numpy as np
from picamera2 import Picamera2
import time
import math

# ---------------- Reference resistor colors (rough HSV, OpenCV ranges) -----------

# You *will* want to tweak these to match your camera.
# H: 0–179, S: 0–255, V: 0–255
REF_COLORS_HSV = {
    "black":  np.array([0,   0,   20 ]),
    "brown":  np.array([10, 180,  80 ]),
    "red":    np.array([0,  220, 200 ]),
    "orange": np.array([10, 220, 220 ]),
    "yellow": np.array([25, 220, 230 ]),
    "green":  np.array([60, 220, 200 ]),
    "blue":   np.array([105,220, 200 ]),
    "violet": np.array([140,220, 180 ]),
    "gray":   np.array([0,   20, 150 ]),
    "white":  np.array([0,   10, 240 ]),
    "gold":   np.array([22,  80, 180 ]),
    "silver": np.array([0,   10, 190 ]),
}

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

# ---------- Color classification ----------

def classify_hsv_to_color(hsv_vec):
    """Return closest resistor color name for given HSV average."""
    best_name = "unknown"
    best_dist = 1e9
    for name, ref in REF_COLORS_HSV.items():
        # simple Euclidean distance in HSV
        d = np.linalg.norm(hsv_vec - ref)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, best_dist

def decode_resistor(colors):
    """
    Decode list of color names into possible resistor values.
    Try both reading directions and 4/5-band interpretations.
    """
    results = []

    def decode_one(seq, direction):
        seq = [c for c in seq if c != "unknown"]
        if len(seq) < 4:
            return

        # 4-band: [d1, d2, mult, tol]
        if all(c in RESISTOR_DIGITS for c in seq[:2]) and seq[2] in RESISTOR_MULTIPLIERS:
            d1 = RESISTOR_DIGITS[seq[0]]
            d2 = RESISTOR_DIGITS[seq[1]]
            mult = RESISTOR_MULTIPLIERS[seq[2]]
            value = (10 * d1 + d2) * mult
            tol = RESISTOR_TOLERANCE.get(seq[3], None)
            results.append((value, tol, direction, 4))

        # 5-band: [d1, d2, d3, mult, tol]
        if len(seq) >= 5 and all(c in RESISTOR_DIGITS for c in seq[:3]) and seq[3] in RESISTOR_MULTIPLIERS:
            d1 = RESISTOR_DIGITS[seq[0]]
            d2 = RESISTOR_DIGITS[seq[1]]
            d3 = RESISTOR_DIGITS[seq[2]]
            mult = RESISTOR_MULTIPLIERS[seq[3]]
            value = (100 * d1 + 10 * d2 + d3) * mult
            tol = RESISTOR_TOLERANCE.get(seq[4], None)
            results.append((value, tol, direction, 5))

    decode_one(colors, "L→R")
    decode_one(list(reversed(colors)), "R→L")
    return results

# ---------- Band detection from one frame ----------

def find_resistor_row(hsv_frame):
    """Pick row with maximum saturation variance along X."""
    sat = hsv_frame[:, :, 1].astype(np.float32)
    var_per_row = sat.var(axis=1)
    return int(np.argmax(var_per_row))

def detect_bands_from_frame(frame_bgr):
    """
    Given a BGR frame, detect color bands.
    Returns: (bands, row_used)
    bands = list of dicts: {xs, xe, color, score}
    """
    hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 1. find best row
    row = find_resistor_row(hsv_full)

    # 2. take a small strip around it
    h, w, _ = frame_bgr.shape
    half_height = 8
    y1 = max(0, row - half_height)
    y2 = min(h, row + half_height)
    strip = hsv_full[y1:y2, :, :]

    # 3. 1D profile: average over Y
    col_mean = strip.mean(axis=0)   # shape: (w, 3)

    # 4. find big changes along X
    diff = np.linalg.norm(np.diff(col_mean, axis=0), axis=1)
    thr = diff.mean() + 1.5 * diff.std()    # tune factor if needed

    segments = []
    start = 0
    min_width = 8

    for x in range(1, w):
        if diff[x - 1] > thr:
            if x - start >= min_width:
                segments.append((start, x))
            start = x
    if w - start >= min_width:
        segments.append((start, w))

    # 5. filter by width and classify colors
    bands = []
    for xs, xe in segments:
        width = xe - xs
        # ignore tiny noise, and very large chunks (background)
        if width < 15 or width > 250:
            continue

        hsv_avg = col_mean[xs:xe].mean(axis=0)
        color, dist = classify_hsv_to_color(hsv_avg)

        bands.append({
            "xs": xs,
            "xe": xe,
            "color": color,
            "score": dist,
        })

    # 6. merge adjacent same-color bands if gap is tiny
    merged = []
    for b in bands:
        if merged and merged[-1]["color"] == b["color"] and b["xs"] - merged[-1]["xe"] < 10:
            merged[-1]["xe"] = b["xe"]
            merged[-1]["score"] = min(merged[-1]["score"], b["score"])
        else:
            merged.append(b)

    return merged, row

# ---------- Main loop ----------

def format_ohms(val):
    if val >= 1_000_000:
        return f"{val / 1_000_000:.2f} MΩ"
    if val >= 1_000:
        return f"{val / 1_000:.2f} kΩ"
    return f"{val:.0f} Ω"

def main():
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)

    print("Press 'q' to quit. Script will continuously estimate resistor value.")

    last_text = "No reading yet"

    while True:
        frame = cam.capture_array()
        display = frame.copy()

        bands, row = detect_bands_from_frame(frame)

        # draw chosen row
        cv2.line(display, (0, row), (display.shape[1], row), (0, 255, 0), 1)

        # list of color names in order
        color_seq = [b["color"] for b in bands]

        # draw bands and names
        for b in bands:
            xs, xe = b["xs"], b["xe"]
            cv2.rectangle(display,
                          (xs, 0),
                          (xe, display.shape[0]),
                          (0, 0, 0),
                          2)
            cv2.putText(display, b["color"],
                        (xs + 2, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

        guesses = decode_resistor(color_seq)
        if guesses:
            # pick the one with smallest tolerance known, or just first
            guesses_sorted = sorted(
                guesses,
                key=lambda g: (g[1] if g[1] is not None else 99)
            )
            value, tol, direction, nband = guesses_sorted[0]
            txt = f"{format_ohms(value)} ({nband}-band, {direction})"
            if tol is not None:
                txt += f" ±{tol}%"
            last_text = txt
        else:
            last_text = "Could not decode"

        cv2.putText(display, last_text,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Resistor", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
