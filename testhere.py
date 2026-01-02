import cv2
import time
import numpy as np
# import cv2
# import numpy as np

# frame = cv2.imread("yellowpurpleblackbrownbrown.png")




# IMAGE_PATH = "yellowpurpleblackbrownbrown.png"
# IMAGE_PATH = "brownblackblackredbrown.png"
# IMAGE_PATH = "greenbrownblackbrownbrown.png"
# IMAGE_PATH = "orangeorangeblackblackbrown.png"
# IMAGE_PATH = "brownblackblackgoldbrown.png"

# Pixelation: smaller -> chunkier, but too small can merge nearby colors.
PIXELATE_SCALE = 0.18

# Use only the middle "core" of the resistor to avoid edge blur/background bleed
CORE_TOP_FRAC = 0.25
CORE_BOT_FRAC = 0.75

# Edge detection on 1D lightness signal (Lab L*)
EDGE_THRESH = 6          # try 5..12 (higher = fewer boundaries)
MIN_BAND_WIDTH = 6       # pixels (increase if you get too many tiny bands)

# If edges are still "soft", increase these:
# - L_SMOOTH_KSIZE (stronger smoothing -> cleaner boundaries)
# - EDGE_DILATE (merge nearby edges)
L_SMOOTH_KSIZE = 11      # odd, try 9, 11, 15, 21
EDGE_DILATE = 3          # try 1..7 (higher merges close boundaries)

# Blue background mask (adjust to your setup)
# rgb(1, 137, 210)
# bgr
# LOWER_BLUE = np.array([180, 120, 0])
# UPPER_BLUE = np.array([230, 150, 10])

# =======================
# HELPERS
# =======================
def smooth_1d(x, ksize):
    """Gaussian smooth a 1D array using OpenCV."""
    x = x.astype(np.float32)
    if ksize % 2 == 0:
        ksize += 1
    x2 = x.reshape(1, -1, 1)  # 1 x W x 1
    x2 = cv2.GaussianBlur(x2, (ksize, 1), 0)
    return x2.reshape(-1)

def group_edges(edge_idx, radius=3):
    """Merge edges that are within +/- radius into a single edge (median)."""
    if len(edge_idx) == 0:
        return []
    edge_idx = np.sort(edge_idx)
    groups = [[edge_idx[0]]]
    for e in edge_idx[1:]:
        if e - groups[-1][-1] <= radius:
            groups[-1].append(e)
        else:
            groups.append([e])
    return [int(np.median(g)) for g in groups]

# =======================
# LOAD
# =======================
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# =======================
# DENOISE + SHARPEN
# =======================
den = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
blur = cv2.GaussianBlur(den, (0, 0), 2.0)
sharp = cv2.addWeighted(den, 1.6, blur, -0.6, 0)

# =======================
# BLOCKIFY (PIXELATE) — local averaging only, avoids global color merging
# =======================
h, w = sharp.shape[:2]
sw, sh = max(1, int(w * PIXELATE_SCALE)), max(1, int(h * PIXELATE_SCALE))
small = cv2.resize(sharp, (sw, sh), interpolation=cv2.INTER_AREA)
blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# =======================
# ROI FIND (NOT BLUE BACKGROUND)
# =======================
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

# =======================
# STEP A: SNAP EACH COLUMN TO ONE COLOR (very sharp vertical steps)
# =======================
snapped = blocky.copy()
for x in range(blocky.shape[1]):
    snapped[:, x] = np.median(blocky[:, x], axis=0)

pre_bil = cv2.bilateralFilter(snapped, 10, 100, 200)
# pre_bil = cv2.bilateralFilter(frame, 10, 10, 200)









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
img = pre_bil

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
# k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
# bg_mask_clean = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN,  k, iterations=MORPH_OPEN_ITERS)
# bg_mask_clean = cv2.morphologyEx(bg_mask_clean, cv2.MORPH_CLOSE, k, iterations=MORPH_CLOSE_ITERS)
# obj_mask_clean = cv2.bitwise_not(bg_mask_clean)








# hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
# lower_blue = np.array([80, 120, 50])
# upper_blue = np.array([130, 255, 255])
# blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
# inverse_mask = cv2.bitwise_not(blue_mask)
inverse_mask = obj_mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
inverse_mask = cv2.dilate(inverse_mask, kernel, iterations=5)
inverse_mask = cv2.erode(inverse_mask, kernel, iterations=8)
cropped = [-1, -1, -1, -1]
last = [0, 0]
file = open("out.txt", "w")
for i in range(len(inverse_mask)):
    for j in range(len(inverse_mask[i])):
        if(cropped[0] == -1 and inverse_mask[i][j] == 0):
            cropped[0] = i
        if(cropped[1] == -1 and inverse_mask[i][j] == 0):
            cropped[1] = j
        if(inverse_mask[i][j] == 255):
            file.write("1")
        else:
            file.write(str(inverse_mask[i][j])) 
            last = [i, j] 
    file.write("\n")
file.close()
cropped[2:] = last
pre_bil_cropped = pre_bil[cropped[0] : cropped[2], cropped[1] : cropped[3]]







inverse_mask = inverse_mask[cropped[0] : cropped[2], cropped[1] : cropped[3]]
oned = False
rectangles = []
file = open("out2.txt", "w")
for j in range(len(inverse_mask[0])):
    if(inverse_mask[0][j] == 0 and oned):
        rectangles[-1].append(0)
        rectangles[-1].append(j)
        oned = False
    elif(inverse_mask[0][j] == 255 and not oned):
        oned = True
        rectangles.append([0, j])
        file.write(str(inverse_mask[0][j]))
file.close()
print(inverse_mask[0][0])
print(rectangles)
averages = []
themask = np.zeros((inverse_mask.__len__(), inverse_mask[0].__len__()), np.uint8)
try:
    for i in rectangles:
        themask [i[0] :len(inverse_mask), (i[1] + int(abs(i[3]-i[1])/2.5)):(i[3] - int(abs(i[3]-i[1])/2.5))] = 1
        print(np.mean(np.mean(pre_bil_cropped[i[0] :len(inverse_mask), (i[1] + int(abs(i[3]-i[1])/2.5)):(i[3] - int(abs(i[3]-i[1])/2.5))], 0), 0))
except Exception as e:
    pass
result = cv2.bitwise_and(pre_bil_cropped, pre_bil_cropped, mask=themask)
cv2.imshow("Display2", frame)
cv2.imshow("Display", result)
cv2.imshow("Display3", pre_bil_cropped)
cv2.imshow("Display4", inverse_mask)
cv2.imshow("Display5", pre_bil)
# cv2.imshow("Display6", pre_bil_cropped)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()