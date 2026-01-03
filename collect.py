import cv2
import time
import numpy as np

# ----------------------------
# REQUIRED: set your image here
# ----------------------------
IMAGE_PATH = "C:\\Users\\kmeng\\OneDrive\\Documents\\GitHub\\resistoridentifier\\images\\Display22.png"  # <-- change this to your filename

PIXELATE_SCALE = 0.18

CORE_TOP_FRAC = 0.25
CORE_BOT_FRAC = 0.75

EDGE_THRESH = 6          # try 5..12 (higher = fewer boundaries)
MIN_BAND_WIDTH = 6       # pixels (increase if you get too many tiny bands)

L_SMOOTH_KSIZE = 11      # odd, try 9, 11, 15, 21
EDGE_DILATE = 3          # try 1..7 (higher merges close boundaries)


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


# -------------------------------------------------------------------
# ADDED: rectangle fallback based on color edges (only used if needed)
# -------------------------------------------------------------------
def rectangles_from_color_edges(pre_bil_img, edge_thresh, smooth_ksize, edge_dilate, min_band_width):
    """
    Produce rectangles like your code expects: [0, start_x, 0, end_x]
    This uses ONLY color transitions across X (no background masking).
    """
    h, w = pre_bil_img.shape[:2]
    if w < 3:
        return []

    # median BGR per column -> shape (1, w, 3)
    col_bgr = np.median(pre_bil_img, axis=0).astype(np.uint8).reshape(1, w, 3)

    # convert to Lab (perceptual-ish spacing) for better edge magnitude
    col_lab = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)[0]  # (w,3)

    # diff between adjacent columns -> (w-1,)
    d = np.abs(col_lab[1:] - col_lab[:-1]).sum(axis=1).astype(np.float32)

    # smooth 1D
    d_s = smooth_1d(d, smooth_ksize)

    # threshold to edges
    edges = (d_s > edge_thresh).astype(np.uint8) * 255  # (w-1,)

    # dilate edges to merge close ones
    edges = cv2.dilate(edges.reshape(1, -1), np.ones((1, edge_dilate), np.uint8), iterations=1).reshape(-1)

    # boundary indices (shift by +1 because edges are between cols)
    edge_idx = (np.where(edges > 0)[0] + 1).astype(int)

    # merge nearby edge indices
    merged = group_edges(edge_idx, radius=3)

    # build segments
    boundaries = [0] + merged + [w]
    rects = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if (b - a) >= min_band_width:
            rects.append([0, int(a), 0, int(b)])

    return rects


frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

den = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
blur = cv2.GaussianBlur(den, (0, 0), 2.0)
sharp = cv2.addWeighted(den, 1.6, blur, -0.6, 0)

h, w = sharp.shape[:2]
sw, sh = max(1, int(w * PIXELATE_SCALE)), max(1, int(h * PIXELATE_SCALE))
small = cv2.resize(sharp, (sw, sh), interpolation=cv2.INTER_AREA)
blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
snapped = blocky.copy()
for x in range(blocky.shape[1]):
    snapped[:, x] = np.median(blocky[:, x], axis=0)

pre_bil = cv2.bilateralFilter(snapped, 10, 100, 200)

DOWNSAMPLE_MAX_W = 320

MIN_S_FOR_DOMINANT = 90      # 0..255
MIN_V_FOR_DOMINANT = 40      # 0..255

H_TOL = 5                   # 0..179 (OpenCV hue)
S_TOL = 70                   # 0..255
V_TOL = 70                   # 0..255

MORPH_K = 5
MORPH_OPEN_ITERS = 1
MORPH_CLOSE_ITERS = 2


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
        valid = np.ones(H.shape, dtype=bool)

    h = H[valid].astype(np.int32)
    s = S[valid].astype(np.int32)
    v = V[valid].astype(np.int32)

    s_bin = np.clip((s * s_bins) // 256, 0, s_bins - 1)
    v_bin = np.clip((v * v_bins) // 256, 0, v_bins - 1)

    idx = h * (s_bins * v_bins) + s_bin * v_bins + v_bin
    hist = np.bincount(idx, minlength=180 * s_bins * v_bins)
    best = int(np.argmax(hist))

    best_h = best // (s_bins * v_bins)
    rem = best % (s_bins * v_bins)
    best_sbin = rem // v_bins
    best_vbin = rem % v_bins

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
        ranges.append((np.array([0,  s0, v0]), np.array([h1,  s1, v1])))
        ranges.append((np.array([180 + h0, s0, v0]), np.array([179, s1, v1])))
    elif h1 > 179:
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


img = pre_bil
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

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
ranges = hsv_range_wrap(dom_h, dom_s, dom_v, H_TOL, S_TOL, V_TOL)
bg_mask = inrange_multi(hsv, ranges)          # 255 = background-like
obj_mask = cv2.bitwise_not(bg_mask)           # 255 = not-background (resistor + others)
inverse_mask = obj_mask

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
inverse_mask = cv2.dilate(inverse_mask, kernel, iterations=5)
inverse_mask = cv2.erode(inverse_mask, kernel, iterations=8)

cropped = [-1, -1, -1, -1]
last = [0, 0]

file = open("out.txt", "w")
for i in range(len(inverse_mask)):
    for j in range(len(inverse_mask[i])):
        if (cropped[0] == -1 and inverse_mask[i][j] == 0):
            cropped[0] = i
        if (cropped[1] == -1 and inverse_mask[i][j] == 0):
            cropped[1] = j
        if (inverse_mask[i][j] == 255):
            file.write("1")
        else:
            file.write(str(inverse_mask[i][j]))
            last = [i, j]
    file.write("\n")
file.close()

cropped[2:] = last

pre_bil_cropped = pre_bil[cropped[0]:cropped[2], cropped[1]:cropped[3]]
inverse_mask = inverse_mask[cropped[0]:cropped[2], cropped[1]:cropped[3]]

# -------------------------------------------------------------------
# Your rectangle detection logic on the first row (kept exactly)
# -------------------------------------------------------------------
oned = False
rectangles = []

file = open("out2.txt", "w")
for j in range(len(inverse_mask[0])):
    if (inverse_mask[0][j] == 0 and oned):
        rectangles[-1].append(0)
        rectangles[-1].append(j)
        oned = False
    elif (inverse_mask[0][j] == 255 and not oned):
        oned = True
        rectangles.append([0, j])
        file.write(str(inverse_mask[0][j]))
file.close()

print(inverse_mask[0][0])
print("mask-rectangles:", rectangles)

# -------------------------------------------------------------------
# ADDED: fallback if mask-based rectangles collapse (e.g., only 1 band)
# -------------------------------------------------------------------
# If your background HSV matches the resistor/bands, the mask can "eat" bands.
# In that case, use pure color-edge segmentation across X on pre_bil_cropped.
if len(rectangles) < 3:
    rect_fb = rectangles_from_color_edges(
        pre_bil_cropped,
        EDGE_THRESH,
        L_SMOOTH_KSIZE,
        EDGE_DILATE,
        MIN_BAND_WIDTH
    )
    print("fallback-rectangles:", rect_fb)
    if len(rect_fb) >= len(rectangles):
        rectangles = rect_fb

averages = []
themask = np.zeros((inverse_mask.__len__(), inverse_mask[0].__len__()), np.uint8)

try:
    for i in rectangles:
        if abs(i[1] - i[3]) < MIN_BAND_WIDTH:
            continue
        themask[
            i[0]:len(inverse_mask),
            (i[1] + int(abs(i[3] - i[1]) / 2.5)):(i[3] - int(abs(i[3] - i[1]) / 2.5))
        ] = 1
        print(np.mean(np.mean(
            pre_bil_cropped[
                i[0]:len(inverse_mask),
                (i[1] + int(abs(i[3] - i[1]) / 2.5)):(i[3] - int(abs(i[3] - i[1]) / 2.5))
            ],
            0
        ), 0))
except Exception as e:
    pass

result = cv2.bitwise_and(pre_bil_cropped, pre_bil_cropped, mask=themask)
cv2.imshow("Display2", frame)
cv2.imshow("Display", result)
cv2.imshow("Display3", pre_bil_cropped)
cv2.imshow("Display4", inverse_mask)
cv2.imshow("Display5", pre_bil)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
