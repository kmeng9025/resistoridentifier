import cv2
import numpy as np
import json
import os

# -------------------------------------------------------------------
# Input image (switch by commenting/uncommenting)
# -------------------------------------------------------------------
# IMAGE_PATH = "yellowpurpleblackbrownbrown.png"
# IMAGE_PATH = "brownblackblackredbrown.png"
# IMAGE_PATH = "greenbrownblackbrownbrown.png"
# IMAGE_PATH = "orangeorangeblackblackbrown.png"
IMAGE_PATH = "brownblackblackgoldbrown.png"

# -------------------------------------------------------------------
# Calibration / detection controls (added previously)
# -------------------------------------------------------------------
MODE = "detect"  # "detect" or "collect"
SAMPLES_JSONL = "band_samples.jsonl"
CALIBRATION_JSON = "calibration.json"
ENHANCE_MODE = "none"  # "none", "contrast", "clahe_Lab"

# -------------------------------------------------------------------
# Existing tuning knobs (kept, even if some are unused)
# -------------------------------------------------------------------
PIXELATE_SCALE = 0.18

CORE_TOP_FRAC = 0.25
CORE_BOT_FRAC = 0.75

EDGE_THRESH = 6       # try 5..12 (higher = fewer boundaries)
MIN_BAND_WIDTH = 6    # pixels (increase if you get too many tiny bands)

L_SMOOTH_KSIZE = 11   # odd, try 9, 11, 15, 21
EDGE_DILATE = 3       # try 1..7 (higher merges close boundaries)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def roi_pixels(pre_bil_cropped, inverse_mask, rect):
    """
    Return the exact ROI pixel array that your original code averages over,
    flattened to Nx3 BGR float32 for calibration/classification.

    rect is one element from rectangles, i.e. [0, start_x, 0, end_x]
    """
    y0 = rect[0]
    x1 = rect[1]
    x2 = rect[3]

    pad = int(abs(x2 - x1) / 2.5)
    xa = (x1 + pad)
    xb = (x2 - pad)

    roi = pre_bil_cropped[y0:len(inverse_mask), xa:xb]
    if roi.size == 0:
        return None
    return roi.reshape(-1, 3).astype(np.float32)


def enhance_for_measurement(pixels_bgr, mode="none"):
    """
    Enhance separation AFTER segmentation, ONLY for color measurement.
    Does NOT affect your mask/rectangles logic at all.

    pixels_bgr: Nx3 float32
    """
    if mode == "none":
        return pixels_bgr

    # Convert to uint8 image-like for simple ops
    p = np.clip(pixels_bgr, 0, 255).astype(np.uint8)
    # reshape to 1xN "image"
    img = p.reshape(1, -1, 3)

    if mode == "clahe_Lab":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2, A, B])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out.reshape(-1, 3).astype(np.float32)

    if mode == "contrast":
        # simple linear contrast stretch around mean
        out = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
        return out.reshape(-1, 3).astype(np.float32)

    return pixels_bgr


def band_feature(pixels_bgr):
    """
    Compute robust features from isolated band pixels.

    Returns:
      - med_bgr: median BGR color
      - chrom: normalized chromaticity (med / sum(med))
      - bright: luminance-ish brightness proxy
    """
    b = pixels_bgr[:, 0]
    g = pixels_bgr[:, 1]
    r = pixels_bgr[:, 2]

    # robust center
    med = np.array([np.median(b), np.median(g), np.median(r)], dtype=np.float32)

    # normalized chromaticity (helps with WB/exposure)
    s = float(med.sum())
    if s <= 1e-6:
        chrom = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        chrom = med / s

    # brightness proxy (helps black vs brown vs yellow)
    bright = float(0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2])

    return {
        "med_bgr": med,
        "chrom": chrom,
        "bright": bright,
    }


def save_sample(sample_path, image_path, band_index, feat, label=""):
    """Append one band sample to a JSONL file for calibration."""
    rec = {
        "image": image_path,
        "band": int(band_index),
        "label": label,
        "med_bgr": feat["med_bgr"].tolist(),
        "chrom": feat["chrom"].tolist(),
        "bright": feat["bright"],
    }
    with open(sample_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_calibration(cal_path):
    with open(cal_path, "r") as f:
        return json.load(f)


def classify_from_calibration(feat, cal):
    """
    Calibration format:
      cal["classes"][name]["chrom_mean"] (len3)
      cal["classes"][name]["chrom_std"]  (len3)
      cal["classes"][name]["bright_mean"]
      cal["classes"][name]["bright_std"]

    Returns: (best_label, confidence, best_score)
    """
    x = np.array(feat["chrom"], dtype=np.float32)
    b = float(feat["bright"])

    best_label = None
    best_score = 1e9
    second_best = 1e9

    for name, c in cal["classes"].items():
        mu = np.array(c["chrom_mean"], dtype=np.float32)
        sd = np.array(c["chrom_std"], dtype=np.float32)
        sd = np.maximum(sd, 1e-4)

        # z-score distance in chromaticity
        z = (x - mu) / sd
        d_chrom = float(np.sqrt((z * z).sum()))

        # brightness z-score (helps brown/yellow/black)
        bmu = float(c["bright_mean"])
        bsd = max(float(c["bright_std"]), 1e-4)
        d_bright = abs(b - bmu) / bsd

        score = d_chrom + 0.6 * d_bright  # weight brightness a bit

        if score < best_score:
            second_best = best_score
            best_score = score
            best_label = name
        elif score < second_best:
            second_best = score

    # confidence from separation
    conf = 1.0 - (best_score / (best_score + second_best + 1e-6))
    conf = float(np.clip(conf, 0.0, 1.0))
    return best_label, conf, best_score


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
# Load image
# -------------------------------------------------------------------
frame_bgr = cv2.imread(IMAGE_PATH)
if frame_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# -------------------------------------------------------------------
# Your preprocessing: denoise -> sharpen -> pixelate -> snap columns -> bilateral
# -------------------------------------------------------------------
denoised = cv2.fastNlMeansDenoisingColored(frame_bgr, None, 5, 5, 7, 21)
blurred = cv2.GaussianBlur(denoised, (0, 0), 2.0)
sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)

img_h, img_w = sharpened.shape[:2]
small_w = max(1, int(img_w * PIXELATE_SCALE))
small_h = max(1, int(img_h * PIXELATE_SCALE))

down = cv2.resize(sharpened, (small_w, small_h), interpolation=cv2.INTER_AREA)
blocky = cv2.resize(down, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

snapped = blocky.copy()
for x in range(blocky.shape[1]):
    snapped[:, x] = np.median(blocky[:, x], axis=0)

pre_bil = cv2.bilateralFilter(snapped, 10, 100, 200)

# -------------------------------------------------------------------
# Background detection in HSV (dominant background color)
# -------------------------------------------------------------------
DOWNSAMPLE_MAX_W = 320

MIN_S_FOR_DOMINANT = 90  # 0..255
MIN_V_FOR_DOMINANT = 40  # 0..255

H_TOL = 5    # 0..179 (OpenCV hue)
S_TOL = 70   # 0..255
V_TOL = 70   # 0..255

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
        ranges.append((np.array([0, s0, v0]), np.array([h1, s1, v1])))
        ranges.append((np.array([180 + h0, s0, v0]), np.array([179, s1, v1])))
    elif h1 > 179:
        ranges.append((np.array([h0, s0, v0]), np.array([179, s1, v1])))
        ranges.append((np.array([0, s0, v0]), np.array([h1 - 180, s1, v1])))
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


img_for_bg = pre_bil
h, w = img_for_bg.shape[:2]

if w > DOWNSAMPLE_MAX_W:
    scale = DOWNSAMPLE_MAX_W / w
    small = cv2.resize(img_for_bg, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
else:
    small = img_for_bg.copy()

hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
dom_h, dom_s, dom_v = dominant_hsv_from_hist(
    hsv_small, min_s=MIN_S_FOR_DOMINANT, min_v=MIN_V_FOR_DOMINANT
)
print("Dominant HSV (OpenCV):", (dom_h, dom_s, dom_v))

hsv_full = cv2.cvtColor(img_for_bg, cv2.COLOR_BGR2HSV)
ranges = hsv_range_wrap(dom_h, dom_s, dom_v, H_TOL, S_TOL, V_TOL)

bg_mask = inrange_multi(hsv_full, ranges)      # 255 = background-like
obj_mask = cv2.bitwise_not(bg_mask)            # 255 = not-background (resistor + others)
inverse_mask = obj_mask

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
inverse_mask = cv2.dilate(inverse_mask, kernel, iterations=5)
inverse_mask = cv2.erode(inverse_mask, kernel, iterations=8)

# -------------------------------------------------------------------
# Your cropping logic (kept exactly)
# -------------------------------------------------------------------
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
print(rectangles)

# -------------------------------------------------------------------
# Your mask fill + mean print logic (kept), with calibration/detection added
# -------------------------------------------------------------------
averages = []
themask = np.zeros((inverse_mask.__len__(), inverse_mask[0].__len__()), np.uint8)

try:
    for rect in rectangles:
        themask[
            rect[0]:len(inverse_mask),
            (rect[1] + int(abs(rect[3] - rect[1]) / 2.5)):(rect[3] - int(abs(rect[3] - rect[1]) / 2.5))
        ] = 1

        # --- Calibration/detection uses the SAME ROI pixels your mean uses ---
        roi_pix = roi_pixels(pre_bil_cropped, inverse_mask, rect)
        if roi_pix is not None:
            roi_pix2 = enhance_for_measurement(roi_pix, mode=ENHANCE_MODE)
            feat = band_feature(roi_pix2)

            if MODE == "collect":
                # You will label later in band_samples.jsonl
                save_sample(SAMPLES_JSONL, IMAGE_PATH, len(averages), feat, label="")
                print("   saved sample ->", SAMPLES_JSONL)

            elif MODE == "detect":
                if os.path.exists(CALIBRATION_JSON):
                    cal = load_calibration(CALIBRATION_JSON)
                    label, conf, score = classify_from_calibration(feat, cal)
                    print(
                        "   ->", label,
                        "conf=", round(conf, 2),
                        "score=", round(score, 2),
                        "chrom=", np.round(feat["chrom"], 3),
                        "bright=", round(feat["bright"], 1),
                    )
                else:
                    print("   (no calibration.json found; set MODE='collect' to gather samples)")

        # YOUR original mean print (kept)
        print(
            np.mean(
                np.mean(
                    pre_bil_cropped[
                        rect[0]:len(inverse_mask),
                        (rect[1] + int(abs(rect[3] - rect[1]) / 2.5)):(rect[3] - int(abs(rect[3] - rect[1]) / 2.5))
                    ],
                    0
                ),
                0
            )
        )
except Exception as e:
    pass

# -------------------------------------------------------------------
# Display outputs (kept)
# -------------------------------------------------------------------
result = cv2.bitwise_and(pre_bil_cropped, pre_bil_cropped, mask=themask)
cv2.imshow("Display2", frame_bgr)
cv2.imshow("Display", result)
cv2.imshow("Display3", pre_bil_cropped)
cv2.imshow("Display4", inverse_mask)
cv2.imshow("Display5", pre_bil)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
