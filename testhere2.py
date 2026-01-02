# # # # import cv2
# # # # import numpy as np

# # # # # ---------- Resistor decoding ----------
# # # # DIGIT = {
# # # #     "black":0,"brown":1,"red":2,"orange":3,"yellow":4,
# # # #     "green":5,"blue":6,"violet":7,"gray":8,"white":9
# # # # }
# # # # MULT = {
# # # #     "silver":1e-2,"gold":1e-1,"black":1,"brown":10,"red":100,"orange":1e3,"yellow":1e4,
# # # #     "green":1e5,"blue":1e6,"violet":1e7,"gray":1e8,"white":1e9
# # # # }
# # # # TOL = {
# # # #     "brown":1,"red":2,"green":0.5,"blue":0.25,"violet":0.1,"gray":0.05,
# # # #     "gold":5,"silver":10
# # # # }

# # # # def decode_resistor(bands):
# # # #     """
# # # #     bands: list like ["brown","black","red","gold"] (4-band)
# # # #            or ["brown","black","black","red","brown"] (5-band)
# # # #     returns (ohms, tolerance_percent)
# # # #     """
# # # #     b = [x.lower() for x in bands]
# # # #     if len(b) == 4:
# # # #         d1, d2, mult, tol = b
# # # #         value = (10*DIGIT[d1] + DIGIT[d2]) * MULT[mult]
# # # #         return value, TOL.get(tol, None)
# # # #     elif len(b) == 5:
# # # #         d1, d2, d3, mult, tol = b
# # # #         value = (100*DIGIT[d1] + 10*DIGIT[d2] + DIGIT[d3]) * MULT[mult]
# # # #         return value, TOL.get(tol, None)
# # # #     else:
# # # #         raise ValueError("Expected 4 or 5 bands")

# # # # def format_ohms(v):
# # # #     if v >= 1e6: return f"{v/1e6:.3g} MΩ"
# # # #     if v >= 1e3: return f"{v/1e3:.3g} kΩ"
# # # #     return f"{v:.3g} Ω"

# # # # # ---------- Rectangle trimming ----------
# # # # def rectified_vertical_rectangles(binary_mask,
# # # #                                  min_area=300,
# # # #                                  min_aspect=2.0,          # height/width
# # # #                                  min_rectangularity=0.75, # contour_area / rect_area
# # # #                                  max_tilt_deg=20):
# # # #     """
# # # #     Returns a mask where each blob is replaced by its best-fit vertical rectangle.
# # # #     This trims 'rectangle + attached blob' into just a rectangle.
# # # #     """
# # # #     m = (binary_mask > 0).astype(np.uint8) * 255
# # # #     out = np.zeros_like(m)

# # # #     cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # #     for c in cnts:
# # # #         area = cv2.contourArea(c)
# # # #         if area < min_area:
# # # #             continue

# # # #         (cx, cy), (w, h), ang = cv2.minAreaRect(c)
# # # #         if w < 1 or h < 1:
# # # #             continue

# # # #         # normalize so h is long side
# # # #         if w > h:
# # # #             w, h = h, w
# # # #             ang += 90

# # # #         aspect = h / w
# # # #         if aspect < min_aspect:
# # # #             continue

# # # #         rect_area = w * h
# # # #         rectangularity = area / rect_area
# # # #         if rectangularity < min_rectangularity:
# # # #             continue

# # # #         # keep near-vertical rectangles
# # # #         # treat 90deg as vertical; accept within max_tilt_deg
# # # #         # normalize angle to [0, 180)
# # # #         a = ang % 180
# # # #         tilt = min(abs(a - 90), abs(a - 270))
# # # #         if tilt > max_tilt_deg:
# # # #             continue

# # # #         box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
# # # #         cv2.fillPoly(out, [box], 255)

# # # #     return out

# # # # frame = cv2.imread("test.png")

# # # # # Faster than heavy bilateral on Pi Zero
# # # # blur = cv2.GaussianBlur(frame, (5, 5), 0)

# # # # hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# # # # lower_blue = np.array([80, 120, 50])
# # # # upper_blue = np.array([130, 255, 255])
# # # # blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# # # # # Your code inverts blue; keep consistent:
# # # # inv = cv2.bitwise_not(blue_mask)

# # # # # Clean mask a bit
# # # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# # # # inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)

# # # # # Fast crop bounding box (replaces your nested loops) :contentReference[oaicite:4]{index=4}
# # # # ys, xs = np.where(inv == 0)  # you were cropping around zeros
# # # # if len(ys) > 0:
# # # #     y0, y1 = ys.min(), ys.max() + 1
# # # #     x0, x1 = xs.min(), xs.max() + 1
# # # #     frame_c = frame[y0:y1, x0:x1]
# # # #     inv_c = inv[y0:y1, x0:x1]
# # # # else:
# # # #     frame_c, inv_c = frame, inv

# # # # # Replace each blob with a vertical rectangle (trims off glint bumps)
# # # # rect_mask = rectified_vertical_rectangles(inv_c,
# # # #                                          min_area=300,
# # # #                                          min_aspect=2.0,
# # # #                                          min_rectangularity=0.75,
# # # #                                          max_tilt_deg=20)

# # # # # OPTIONAL: extract each rectangle ROI and compute average color per band
# # # # cnts, _ = cv2.findContours(rect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # # boxes = []
# # # # for c in cnts:
# # # #     (cx, cy), (w, h), ang = cv2.minAreaRect(c)
# # # #     boxes.append((cx, cy, w, h, ang))

# # # # # Sort left-to-right (typical band order)
# # # # boxes.sort(key=lambda t: t[0])

# # # # band_bgr = []
# # # # for (cx, cy, w, h, ang) in boxes:
# # # #     box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
# # # #     m = np.zeros(rect_mask.shape, np.uint8)
# # # #     cv2.fillPoly(m, [box], 255)
# # # #     pixels = frame_c[m == 255]
# # # #     if len(pixels) == 0:
# # # #         continue
# # # #     bgr = np.median(pixels, axis=0)  # median is robust to glare
# # # #     band_bgr.append(tuple(map(float, bgr)))

# # # # # TODO: map band_bgr -> band color names using your classifier
# # # # # For now, if you already have them:
# # # # bands = ["brown", "black", "red", "gold"]  # <-- replace with your detected band names
# # # # ohms, tol = decode_resistor(bands)
# # # # print("Bands:", bands)
# # # # print("Value:", format_ohms(ohms), (f"±{tol}%" if tol is not None else ""))

# # # # # Visualize
# # # # result = cv2.bitwise_and(frame_c, frame_c, mask=rect_mask)
# # # # cv2.imshow("Rect-mask", rect_mask)
# # # # cv2.imshow("Rectified result", result)
# # # # cv2.waitKey(0)
# # # # cv2.destroyAllWindows()
# # # import cv2, numpy as np

# # # img = cv2.imread("yellowpurpleblackbrownbrown.png")

# # # # denoise + sharpen
# # # den = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
# # # blur = cv2.GaussianBlur(den, (0,0), 2.0)
# # # sharp = cv2.addWeighted(den, 1.6, blur, -0.6, 0)

# # # # pixelate (blockify without global color merging)
# # # h, w = sharp.shape[:2]
# # # scale = 0.18
# # # small = cv2.resize(sharp, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
# # # blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# # # # use Lab for more stable color measures
# # # lab = cv2.cvtColor(blocky, cv2.COLOR_BGR2LAB)
# # # L, A, B = cv2.split(lab)

# # # cv2.imshow("orig", img)
# # # cv2.imshow("sharp", sharp)
# # # cv2.imshow("blocky", blocky)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
# # import cv2
# # import numpy as np

# # # ---------- CONFIG ----------
# # IMAGE_PATH = "yellowpurpleblackbrownbrown.png"

# # # Blockify strength (smaller = chunkier, but too small will over-average)
# # PIXELATE_SCALE = 0.18

# # # "Core" crop to avoid top/bottom edge contamination from blur/background
# # CORE_TOP_FRAC = 0.25
# # CORE_BOT_FRAC = 0.75

# # # 1D smoothing along x on the per-column color signal (odd width)
# # SMOOTH_KSIZE = 9  # try 7, 9, 11, 15

# # # ---------- LOAD ----------
# # frame = cv2.imread(IMAGE_PATH)
# # if frame is None:
# #     raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# # # ---------- DENOISE + SHARPEN ----------
# # den = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
# # blur = cv2.GaussianBlur(den, (0, 0), 2.0)
# # sharp = cv2.addWeighted(den, 1.6, blur, -0.6, 0)

# # # ---------- BLOCKIFY (PIXELATE) ----------
# # h, w = sharp.shape[:2]
# # sw, sh = max(1, int(w * PIXELATE_SCALE)), max(1, int(h * PIXELATE_SCALE))
# # small = cv2.resize(sharp, (sw, sh), interpolation=cv2.INTER_AREA)
# # blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# # # ---------- OPTIONAL: FIND ROI (IF YOU ALREADY HAVE ROI, SKIP THIS) ----------
# # # If your resistor is already cropped tightly, you can skip this section.
# # # This uses your earlier idea: "object = not blue background".
# # hsv = cv2.cvtColor(blocky, cv2.COLOR_BGR2HSV)
# # lower_blue = np.array([80, 120, 50])
# # upper_blue = np.array([130, 255, 255])
# # blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
# # obj = cv2.bitwise_not(blue_mask)

# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# # obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel, iterations=2)
# # obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN,  kernel, iterations=1)

# # ys, xs = np.where(obj == 0)  # 0 = object (not blue)
# # if len(xs) == 0:
# #     raise RuntimeError("No object found. Adjust HSV blue thresholds.")

# # y0, y1 = ys.min(), ys.max()
# # x0, x1 = xs.min(), xs.max()

# # roi = blocky[y0:y1+1, x0:x1+1].copy()

# # # ---------- CROP TO CORE (REMOVE TOP/BOTTOM EDGES) ----------
# # rh = roi.shape[0]
# # yt = int(CORE_TOP_FRAC * rh)
# # yb = int(CORE_BOT_FRAC * rh)
# # core = roi[yt:yb, :].copy()

# # # ---------- SNAP EACH COLUMN TO A SINGLE COLOR (SHARP VERTICAL EDGES) ----------
# # snapped = core.copy()
# # for x in range(core.shape[1]):
# #     snapped[:, x] = np.median(core[:, x], axis=0)

# # # ---------- REMOVE VERTICAL NOISE (SMOOTH ALONG X ON COLOR SIGNAL) ----------
# # # Compute mean BGR per column (shape: [W, 3])
# # col_means = np.mean(snapped, axis=0).astype(np.float32)

# # # Smooth along x only: treat this as a 1xW "image" with 3 channels
# # if SMOOTH_KSIZE % 2 == 0:
# #     SMOOTH_KSIZE += 1
# # col_means_smooth = cv2.GaussianBlur(col_means.reshape(1, -1, 3), (SMOOTH_KSIZE, 1), 0).reshape(-1, 3)

# # # Rebuild the image from the smoothed per-column colors
# # final = np.zeros_like(snapped, dtype=np.uint8)
# # for x in range(final.shape[1]):
# #     final[:, x] = np.clip(col_means_smooth[x], 0, 255).astype(np.uint8)

# # # ---------- DISPLAY ----------
# # cv2.imshow("Original", frame)
# # cv2.imshow("Denoised+Sharpened", sharp)
# # cv2.imshow("Blocky (Pixelated)", blocky)
# # cv2.imshow("ROI", roi)
# # cv2.imshow("Core (edges removed)", core)
# # cv2.imshow("Snapped (sharp band edges)", snapped)
# # cv2.imshow("Final (vertical noise reduced)", final)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# import cv2
# import numpy as np

# # =======================
# # CONFIG (tune these)
# # =======================
# IMAGE_PATH = "yellowpurpleblackbrownbrown.png"

# # Pixelation: smaller -> chunkier, but too small can merge nearby colors.
# PIXELATE_SCALE = 0.18

# # Use only the middle "core" of the resistor to avoid edge blur/background bleed
# CORE_TOP_FRAC = 0.25
# CORE_BOT_FRAC = 0.75

# # Edge detection on 1D lightness signal (Lab L*)
# EDGE_THRESH = 6          # try 5..12 (higher = fewer boundaries)
# MIN_BAND_WIDTH = 6       # pixels (increase if you get too many tiny bands)

# # If edges are still "soft", increase these:
# # - L_SMOOTH_KSIZE (stronger smoothing -> cleaner boundaries)
# # - EDGE_DILATE (merge nearby edges)
# L_SMOOTH_KSIZE = 11      # odd, try 9, 11, 15, 21
# EDGE_DILATE = 3          # try 1..7 (higher merges close boundaries)

# # Blue background mask (adjust to your setup)
# LOWER_BLUE = np.array([80, 120, 50])
# UPPER_BLUE = np.array([130, 255, 255])

# # =======================
# # HELPERS
# # =======================
# def smooth_1d(x, ksize):
#     """Gaussian smooth a 1D array using OpenCV."""
#     x = x.astype(np.float32)
#     if ksize % 2 == 0:
#         ksize += 1
#     x2 = x.reshape(1, -1, 1)  # 1 x W x 1
#     x2 = cv2.GaussianBlur(x2, (ksize, 1), 0)
#     return x2.reshape(-1)

# def group_edges(edge_idx, radius=3):
#     """Merge edges that are within +/- radius into a single edge (median)."""
#     if len(edge_idx) == 0:
#         return []
#     edge_idx = np.sort(edge_idx)
#     groups = [[edge_idx[0]]]
#     for e in edge_idx[1:]:
#         if e - groups[-1][-1] <= radius:
#             groups[-1].append(e)
#         else:
#             groups.append([e])
#     return [int(np.median(g)) for g in groups]

# # =======================
# # LOAD
# # =======================
# frame = cv2.imread(IMAGE_PATH)
# if frame is None:
#     raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# # =======================
# # DENOISE + SHARPEN
# # =======================
# den = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
# blur = cv2.GaussianBlur(den, (0, 0), 2.0)
# sharp = cv2.addWeighted(den, 1.6, blur, -0.6, 0)

# # =======================
# # BLOCKIFY (PIXELATE) — local averaging only, avoids global color merging
# # =======================
# h, w = sharp.shape[:2]
# sw, sh = max(1, int(w * PIXELATE_SCALE)), max(1, int(h * PIXELATE_SCALE))
# small = cv2.resize(sharp, (sw, sh), interpolation=cv2.INTER_AREA)
# blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# # =======================
# # ROI FIND (NOT BLUE BACKGROUND)
# # =======================
# hsv = cv2.cvtColor(blocky, cv2.COLOR_BGR2HSV)
# blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
# obj = cv2.bitwise_not(blue_mask)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel, iterations=2)
# obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN,  kernel, iterations=1)

# ys, xs = np.where(obj == 0)  # object pixels (not blue)
# if len(xs) == 0:
#     raise RuntimeError("No object found. Adjust BLUE HSV thresholds.")

# y0, y1 = ys.min(), ys.max()
# x0, x1 = xs.min(), xs.max()

# roi = blocky[y0:y1+1, x0:x1+1].copy()

# # =======================
# # CROP TO CORE (REMOVE TOP/BOTTOM EDGES)
# # =======================
# rh = roi.shape[0]
# yt = int(CORE_TOP_FRAC * rh)
# yb = int(CORE_BOT_FRAC * rh)
# core = roi[yt:yb, :].copy()

# # =======================
# # STEP A: SNAP EACH COLUMN TO ONE COLOR (very sharp vertical steps)
# # =======================
# snapped = core.copy()
# for x in range(core.shape[1]):
#     snapped[:, x] = np.median(core[:, x], axis=0)

# snapped = cv2.bilateralFilter(snapped, 10, 100, 200)
# # =======================
# # SHOW
# # =======================
# cv2.imshow("Original", frame)
# cv2.imshow("Blocky", blocky)
# cv2.imshow("ROI", roi)
# cv2.imshow("Core (edges removed)", core)
# cv2.imshow("Snapped (col median)", snapped)
# # cv2.imshow("Final (razor band edges)", final)
# # cv2.imshow("Final + boundaries", debug)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np

# =======================
# CONFIG
# =======================
IMAGE_PATH = "yellowpurpleblackbrownbrown.png"

# How much to downsample for speed (dominant-color estimation only)
DOWNSAMPLE_MAX_W = 320

# Ignore near-gray pixels when searching for the dominant "background color"
# (background blue is usually high saturation; resistor bands often include low sat colors)
MIN_S_FOR_DOMINANT = 90      # 0..255
MIN_V_FOR_DOMINANT = 40      # 0..255

# HSV range half-widths around the dominant background color
# Hue wrap-around is handled automatically.
H_TOL = 5                   # 0..179 (OpenCV hue)
S_TOL = 70                   # 0..255
V_TOL = 70                   # 0..255

# Morphology cleanup
MORPH_K = 5
MORPH_OPEN_ITERS = 1
MORPH_CLOSE_ITERS = 2

# =======================
# HELPERS
# =======================
def dominant_hsv_from_hist(hsv_img, min_s=60, min_v=40, h_bins=180, s_bins=64, v_bins=64):
    """
    Find the most common HSV color (mode) using a 3D histogram, ignoring low-sat/low-val pixels.
    Returns (h, s, v) in OpenCV HSV ranges: H 0..179, S 0..255, V 0..255.
    """
    H = hsv_img[:, :, 0]
    S = hsv_img[:, :, 1]
    V = hsv_img[:, :, 2]

    valid = (S >= min_s) & (V >= min_v)
    if np.count_nonzero(valid) < 100:
        # Fallback: don't filter if too few pixels pass
        valid = np.ones(H.shape, dtype=bool)

    h = H[valid].astype(np.int32)
    s = S[valid].astype(np.int32)
    v = V[valid].astype(np.int32)

    # Bin S and V to reduce noise and speed up the 3D histogram
    s_bin = np.clip((s * s_bins) // 256, 0, s_bins - 1)
    v_bin = np.clip((v * v_bins) // 256, 0, v_bins - 1)

    # 3D histogram indexing: idx = h*(s_bins*v_bins) + s_bin*v_bins + v_bin
    idx = h * (s_bins * v_bins) + s_bin * v_bins + v_bin
    hist = np.bincount(idx, minlength=180 * s_bins * v_bins)
    best = int(np.argmax(hist))

    best_h = best // (s_bins * v_bins)
    rem = best % (s_bins * v_bins)
    best_sbin = rem // v_bins
    best_vbin = rem % v_bins

    # Convert bin centers back to S,V (0..255)
    best_s = int((best_sbin + 0.5) * 256 / s_bins)
    best_v = int((best_vbin + 0.5) * 256 / v_bins)

    return best_h, best_s, best_v

def hsv_range_wrap(h, s, v, h_tol, s_tol, v_tol):
    """
    Build one or two HSV ranges (lower/upper) handling hue wrap-around for OpenCV hue [0..179].
    Returns a list of (lower, upper) np.array pairs.
    """
    s0 = max(0, s - s_tol)
    s1 = min(255, s + s_tol)
    v0 = max(0, v - v_tol)
    v1 = min(255, v + v_tol)

    h0 = h - h_tol
    h1 = h + h_tol

    ranges = []
    if h0 < 0:
        # Wrap below 0: [0..h1] and [180+h0..179]
        ranges.append((np.array([0,  s0, v0]), np.array([h1,  s1, v1])))
        ranges.append((np.array([180 + h0, s0, v0]), np.array([179, s1, v1])))
    elif h1 > 179:
        # Wrap above 179: [h0..179] and [0..h1-180]
        ranges.append((np.array([h0, s0, v0]), np.array([179, s1, v1])))
        ranges.append((np.array([0,  s0, v0]), np.array([h1 - 180, s1, v1])))
    else:
        ranges.append((np.array([h0, s0, v0]), np.array([h1, s1, v1])))

    return ranges

def inrange_multi(hsv, ranges):
    """Apply cv2.inRange for multiple ranges and OR them together."""
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask

# =======================
# LOAD
# =======================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# =======================
# DOWNSAMPLE for dominant-color estimation
# =======================
h, w = img.shape[:2]
if w > DOWNSAMPLE_MAX_W:
    scale = DOWNSAMPLE_MAX_W / w
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
else:
    small = img.copy()

hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
dom_h, dom_s, dom_v = dominant_hsv_from_hist(
    hsv_small, min_s=MIN_S_FOR_DOMINANT, min_v=MIN_V_FOR_DOMINANT
)

print("Dominant HSV (OpenCV):", (dom_h, dom_s, dom_v))

# =======================
# Build "background blue" mask from dominant HSV
# =======================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ranges = hsv_range_wrap(dom_h, dom_s, dom_v, H_TOL, S_TOL, V_TOL)
bg_mask = inrange_multi(hsv, ranges)          # 255 = background-like
obj_mask = cv2.bitwise_not(bg_mask)           # 255 = not-background (resistor + others)

# Cleanup masks
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
bg_mask_clean = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN,  k, iterations=MORPH_OPEN_ITERS)
bg_mask_clean = cv2.morphologyEx(bg_mask_clean, cv2.MORPH_CLOSE, k, iterations=MORPH_CLOSE_ITERS)
obj_mask_clean = cv2.bitwise_not(bg_mask_clean)

# Visualize background selection
bg_only = cv2.bitwise_and(img, img, mask=bg_mask_clean)
obj_only = cv2.bitwise_and(img, img, mask=obj_mask_clean)

cv2.imshow("Original", img)
cv2.imshow("Background mask (dominant HSV)", bg_mask_clean)
cv2.imshow("Background only", bg_only)
cv2.imshow("Object (not background)", obj_only)

cv2.waitKey(0)
cv2.destroyAllWindows()
