# import cv2



import cv2 as cv

#  establishing the colour ranges to detect the colour on the resistor.
#  these colour values vary depending on the camera settings, white balance and lighting.
#  Vary these parameters to suit your use-case
#  Colours are thresholded in the HSV colour space. more can be found at (https://en.wikipedia.org/wiki/HSL_and_HSV)
Colour_Range = [
    [(0, 0, 0), (255, 255, 20), "BLACK", 0, (0, 0, 0)],
    [(0, 90, 10), (15, 250, 100), "BROWN", 1, (0, 51, 102)],
    [(0, 30, 80), (10, 255, 200), "RED", 2, (0, 0, 255)],
    [(5, 150, 150), (15, 235, 250), "ORANGE", 3, (0, 128, 255)],  # ok
    [(50, 100, 100), (70, 255, 255), "YELLOW", 4, (0, 255, 255)],
    [(45, 100, 50), (75, 255, 255), "GREEN", 5, (0, 255, 0)],  # ok
    [(100, 150, 0), (140, 255, 255), "BLUE", 6, (255, 0, 0)],  # ok
    [(120, 40, 100), (140, 250, 220), "VIOLET", 7, (255, 0, 127)],
    [(0, 0, 50), (179, 50, 80), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 90), (179, 15, 250), "WHITE", 9, (255, 255, 255)],
]

Red_top_low = (160, 30, 80)
Red_top_high = (179, 255, 200)

# setting up other basic necessities such as font and minimum area for a valid contour #
min_area = 0  # this parameter is determined after testing on various images
FONT = cv.FONT_HERSHEY_SIMPLEX


# method to find bands of the resistor
def findBands(img):
    img1 = cv.bilateralFilter(img, 40, 90, 90)  # image is bilaterally filtered to remove noise
    img_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # image then converted to greyscale for thresholding
    img_hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)  # image is converted to HSV colourspace for colour selection
    thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 79,
                                  2)  # adaptive threshold is used to filter out the background
    thresh = cv.bitwise_not(thresh)

    bandpos = []

    for clr in Colour_Range:  # check with the pre- defined colour spaces
        mask = cv.inRange(img_hsv, clr[0], clr[1])
        if clr[2] == 'RED':  # creates two masks for the colour red a it has two colour bounds
            red_mask = cv.inRange(img_hsv, Red_top_low, Red_top_high)
            mask = cv.bitwise_or(red_mask, mask, mask)
        mask = cv.bitwise_and(mask, thresh, mask=mask)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours) - 1, -1, -1):
            if validContours(contours[i]):
                lmp = tuple(
                    contours[i][contours[i][:, :, 0].argmin()][0])  # finds the left most point of each valid contour
                bandpos += [lmp + tuple(clr[2:])]
            else:
                contours.pop(i)
        cv.drawContours(img1, contours, -1, clr[-1], 3)  # draws contours on screen

    cv.imshow('Contour Display', img1)
    return sorted(bandpos,
                  key=lambda tup: tup[0])  # returns a list of valid contours sorted by least value of leftmost point


# method to check the validity of the contours
def validContours(cont):
    if cv.contourArea(cont) < min_area:  # filters out all the tiny contours
        return False
    else:
        x, y, w, h = cv.boundingRect(cont)
        if float(w) / h > 0.40:
            return False
    return True


def displayResults(sortedbands):
    strvalue = ""
    if len(sortedbands) in [3, 4, 5]:
        for band in sortedbands[:-1]:
            strvalue += str(band[3])  # calculates the value of resistance
        intvalue = int(strvalue)
        intvalue *= 10 ** sortedbands[-1][3]  # applies the correct multiplier and stores the final resistance value
        print("The Resistance is ", intvalue, "ohms")
    return


# main method, here we accept the image.
if __name__ == '__main__':
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
        findBands(frame)
        # cv.imshow("Display", frame)
        time.sleep(2)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cam.stop()
    cv.destroyAllWindows()
# import cv2
# import numpy as np
# from picamera2 import Picamera2
# import time

# # ---------- Color helpers ----------

# # Rough mapping color name -> digit/tolerance
# RESISTOR_DIGITS = {
#     "black": 0,
#     "brown": 1,
#     "red": 2,
#     "orange": 3,
#     "yellow": 4,
#     "green": 5,
#     "blue": 6,
#     "violet": 7,
#     "gray": 8,
#     "white": 9,
# }

# RESISTOR_MULTIPLIERS = {
#     "black": 1,
#     "brown": 10,
#     "red": 100,
#     "orange": 1_000,
#     "yellow": 10_000,
#     "green": 100_000,
#     "blue": 1_000_000,
#     "violet": 10_000_000,
#     "gray": 100_000_000,
#     "white": 1_000_000_000,
#     "gold": 0.1,
#     "silver": 0.01,
# }

# RESISTOR_TOLERANCE = {
#     "brown": 1,
#     "red": 2,
#     "green": 0.5,
#     "blue": 0.25,
#     "violet": 0.1,
#     "gray": 0.05,
#     "gold": 5,
#     "silver": 10,
# }


# def classify_color(hsv):
#     """Very rough HSV -> resistor color name classifier.
#        You WILL want to tweak thresholds for your lighting.
#     """
#     h, s, v = hsv

#     # black / white / gray first
#     if v < 40:
#         return "black"
#     if s < 40 and v > 200:
#         return "white"
#     if s < 40:
#         return "gray"

#     # Brown is basically dark orange
#     if 8 < h < 30 and v < 120:
#         return "brown"

#     # Gold / silver (low saturation, mid value) – rarely needed
#     if s < 80 and 120 < v < 200:
#         # more yellow-ish
#         if 15 < h < 45:
#             return "gold"
#         else:
#             return "silver"

#     # Hue-based colors (0-179 in OpenCV)
#     if h < 8 or h > 170:
#         return "red"
#     if 8 <= h < 25:
#         return "orange"
#     if 25 <= h < 35:
#         return "yellow"
#     if 35 <= h < 85:
#         return "green"
#     if 85 <= h < 130:
#         return "blue"
#     if 130 <= h < 160:
#         return "violet"

#     return "unknown"


# def decode_resistor(colors):
#     """
#     Interpret a list of color names as a 4- or 5-band resistor.
#     Try both directions (because we may not know which end is tolerance).
#     Returns a list of possible (value_ohms, tolerance_percent, direction).
#     """
#     results = []

#     def decode_one(seq, direction):
#         # strip "unknown"
#         seq = [c for c in seq if c != "unknown"]
#         if len(seq) < 4:
#             return

#         # assume 4-band: [d1, d2, multiplier, tolerance]
#         c = seq[:4]
#         if all(col in RESISTOR_DIGITS for col in c[:2]) and c[2] in RESISTOR_MULTIPLIERS:
#             d1 = RESISTOR_DIGITS[c[0]]
#             d2 = RESISTOR_DIGITS[c[1]]
#             mult = RESISTOR_MULTIPLIERS[c[2]]
#             value = (10 * d1 + d2) * mult
#             tol = RESISTOR_TOLERANCE.get(c[3], None)
#             results.append((value, tol, direction))

#         # assume 5-band: [d1, d2, d3, multiplier, tolerance]
#         if len(seq) >= 5:
#             c = seq[:5]
#             if all(col in RESISTOR_DIGITS for col in c[:3]) and c[3] in RESISTOR_MULTIPLIERS:
#                 d1 = RESISTOR_DIGITS[c[0]]
#                 d2 = RESISTOR_DIGITS[c[1]]
#                 d3 = RESISTOR_DIGITS[c[2]]
#                 mult = RESISTOR_MULTIPLIERS[c[3]]
#                 value = (100 * d1 + 10 * d2 + d3) * mult
#                 tol = RESISTOR_TOLERANCE.get(c[4], None)
#                 results.append((value, tol, direction))

#     decode_one(colors, "L→R")
#     decode_one(list(reversed(colors)), "R→L")
#     return results


# # ---------- Band detection from a frame ----------

# def find_bands(frame):
#     """
#     Given a BGR frame, find color bands along the resistor.
#     Returns: list of (x_start, x_end, color_name)
#     """

#     h, w, _ = frame.shape

#     # Crop a horizontal strip where the resistor is.
#     # You may need to tune these values depending on slit position.
#     y1 = int(h * 0.55)
#     y2 = int(h * 0.75)
#     strip = frame[y1:y2, :]

#     # Smooth and convert to HSV
#     blur = cv2.GaussianBlur(strip, (9, 9), 0)
#     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

#     # Average each column over Y -> get 1D color profile (w x 3)
#     col_mean = hsv.mean(axis=0)

#     # Look at color differences between neighbouring columns
#     diff = np.linalg.norm(np.diff(col_mean, axis=0), axis=1)

#     # Threshold for "big change" – tune this!
#     # using mean + some factor of std is a bit more adaptive
#     thr = diff.mean() + 1.5 * diff.std()

#     segments = []
#     start = 0
#     min_width = 5  # ignore very tiny segments (noise)

#     for x in range(1, w):
#         if diff[x - 1] > thr:
#             if x - start >= min_width:
#                 segments.append((start, x))
#             start = x
#     if w - start >= min_width:
#         segments.append((start, w))

#     # Compute avg color for each segment and classify
#     bands = []
#     for (xs, xe) in segments:
#         seg_hsv = col_mean[xs:xe].mean(axis=0)
#         color = classify_color(seg_hsv)
#         width = xe - xs

#         # Heuristic: ignore very wide segments (likely body or background)
#         # and very narrow ones (noise). You can adjust these.
#         if width < 15 or width > 250:
#             continue

#         bands.append((xs, xe, color))

#     # Optional: merge consecutive segments of same color
#     merged = []
#     for xs, xe, color in bands:
#         if merged and merged[-1][2] == color and xs - merged[-1][1] < 10:
#             # extend last segment
#             last_xs, last_xe, last_color = merged[-1]
#             merged[-1] = (last_xs, xe, last_color)
#         else:
#             merged.append((xs, xe, color))

#     return merged


# # ---------- Main camera loop ----------

# def main():
#     cam = Picamera2()
#     cfg = cam.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
#     cam.configure(cfg)
#     cam.start()
#     time.sleep(2)

#     print("Press 'q' to quit, 's' to force a reading.")
#     last_print = 0

#     while True:
#         frame = cam.capture_array()
#         display = frame.copy()

#         # Find bands every ~0.5 s
#         now = time.time()
#         do_read = (now - last_print > 0.5)

#         bands = find_bands(frame)

#         # Draw bands on the display frame
#         for xs, xe, color in bands:
#             cv2.rectangle(display, (xs, 0), (xe, display.shape[0]), (0, 0, 0), 2)
#             cv2.putText(display, color, (xs + 2, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

#         if do_read and bands:
#             last_print = now
#             color_seq = [c for _, _, c in bands]
#             print("Detected band colors:", color_seq)
#             guesses = decode_resistor(color_seq)

#             if guesses:
#                 for value, tol, direction in guesses:
#                     if value >= 1_000_000:
#                         txt = f"{value/1_000_000:.2f} MΩ"
#                     elif value >= 1_000:
#                         txt = f"{value/1_000:.2f} kΩ"
#                     else:
#                         txt = f"{value:.0f} Ω"
#                     if tol is not None:
#                         txt += f" ±{tol}%"
#                     print(f"  {direction}: {txt}")
#             else:
#                 print("  Could not decode a valid resistor from colors.")

#         cv2.imshow("Resistor", display)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         if key == ord("s"):
#             last_print = 0  # force immediate read

#     cam.stop()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
