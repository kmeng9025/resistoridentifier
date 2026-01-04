import cv2
import numpy as np
import time
from picamera2 import Picamera2
import LCD1602

FRAME_INTERVAL_SEC = 0.5

DEBUG_SAVE = False
DEBUG_DIR = "debug_frames"
DEBUG_EVERY_N = 10

CALIBRATION = {
    "classes": {
        "br": {
            "chrom_mean": [0.4798282980918884, 0.3002983629703522, 0.21987439692020416],
            "chrom_std":  [0.03936861455440521, 0.01662791520357132, 0.048419173806905746],
            "bright_mean": 66.7628402709961,
            "bright_std": 5.724446773529053
        },
        "bl": {
            "chrom_mean": [0.6021039485931396, 0.3171629011631012, 0.08073320984840393],
            "chrom_std":  [0.0224063228815794, 0.007957104593515396, 0.027250956743955612],
            "bright_mean": 50.54780197143555,
            "bright_std": 3.9904885292053223
        },
        "grn": {
            "chrom_mean": [0.4440917372703552, 0.5413482189178467, 0.014560043811798096],
            "chrom_std":  [0.01433463953435421, 0.024645671248435974, 0.01157507486641407],
            "bright_mean": 88.10596466064453,
            "bright_std": 10.14437198638916
        },
        "r": {
            "chrom_mean": [0.3880232870578766, 0.1856708973646164, 0.42630574107170105],
            "chrom_std":  [0.015979357063770294, 0.014997826889157295, 0.0298507921397686],
            "bright_mean": 76.12236785888672,
            "bright_std": 3.514045238494873
        },
        "pur": {
            "chrom_mean": [0.7752211093902588, 0.21175555884838104, 0.013023421168327332],
            "chrom_std":  [0.021297158673405647, 0.014256240800023079, 0.009865746833384037],
            "bright_mean": 49.2727165222168,
            "bright_std": 1.1102464199066162
        },
        "ylw": {
            "chrom_mean": [0.41115206480026245, 0.4983845055103302, 0.09046328812837601],
            "chrom_std":  [0.036821361631155014, 0.02131418138742447, 0.03309009224176407],
            "bright_mean": 105.08103942871094,
            "bright_std": 11.39084529876709
        },
        "gld": {
            "chrom_mean": [0.40534672141075134, 0.3510783612728119, 0.24357491731643677],
            "chrom_std":  [0.02350943349301815, 0.006786972284317017, 0.02650662139058113],
            "bright_mean": 146.63255310058594,
            "bright_std": 11.007163047790527
        },
        "org": {
            "chrom_mean": [0.3136765956878662, 0.2967498004436493, 0.38957351446151733],
            "chrom_std":  [0.03427901491522789, 0.00971413403749466, 0.04143955931067467],
            "bright_mean": 108.69397735595703,
            "bright_std": 10.307095527648926
        }
    }
}

PIXELATE_SCALE = 0.18

DOWNSAMPLE_MAX_W = 320
MIN_S_FOR_DOMINANT = 90
MIN_V_FOR_DOMINANT = 40

H_TOL = 5
S_TOL = 70
V_TOL = 70

# ---- cached kernel: do NOT rebuild per frame ----
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def band_feature_from_roi(roi_bgr_uint8):
    # Avoid roi.reshape(-1, 3) temporary: compute channel medians directly
    b = np.median(roi_bgr_uint8[..., 0])
    g = np.median(roi_bgr_uint8[..., 1])
    r = np.median(roi_bgr_uint8[..., 2])
    med = np.array([b, g, r], dtype=np.float32)

    s = float(med.sum())
    chrom = (med / s) if s > 1e-6 else np.array([0.0, 0.0, 0.0], dtype=np.float32)

    bright = float(0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2])
    return {"med_bgr": med, "chrom": chrom, "bright": bright}


def classify_from_calibration(feat, cal):
    x = np.array(feat["chrom"], dtype=np.float32)
    b = float(feat["bright"])

    best_label = None
    best_score = 1e9
    second_best = 1e9

    for name, c in cal["classes"].items():
        mu = np.array(c["chrom_mean"], dtype=np.float32)
        sd = np.maximum(np.array(c["chrom_std"], dtype=np.float32), 1e-4)

        z = (x - mu) / sd
        d_chrom = float(np.sqrt((z * z).sum()))

        bmu = float(c["bright_mean"])
        bsd = max(float(c["bright_std"]), 1e-4)
        d_bright = abs(b - bmu) / bsd

        score = d_chrom + 0.6 * d_bright

        if score < best_score:
            second_best = best_score
            best_score = score
            best_label = name
        elif score < second_best:
            second_best = score

    conf = 1.0 - (best_score / (best_score + second_best + 1e-6))
    conf = float(np.clip(conf, 0.0, 1.0))
    return best_label, conf, best_score


def dominant_hsv_from_hist(hsv_img, min_s=60, min_v=40, h_bins=180, s_bins=64, v_bins=64):
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
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def process_frame(frame_bgr, frame_idx=0):
    # --- Preprocess (ops + args unchanged) ---
    denoised = cv2.fastNlMeansDenoisingColored(frame_bgr, None, 5, 5, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)

    img_h, img_w = sharpened.shape[:2]
    small_w = max(1, int(img_w * PIXELATE_SCALE))
    small_h = max(1, int(img_h * PIXELATE_SCALE))

    down = cv2.resize(sharpened, (small_w, small_h), interpolation=cv2.INTER_AREA)
    blocky = cv2.resize(down, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    snapped = blocky.copy()
    snapped[:, :] = np.median(blocky, axis=0)

    pre_bil = cv2.bilateralFilter(snapped, 10, 100, 200)

    # --- Dominant background HSV selection (unchanged) ---
    h0, w0 = pre_bil.shape[:2]
    if w0 > DOWNSAMPLE_MAX_W:
        scale = DOWNSAMPLE_MAX_W / w0
        small = cv2.resize(pre_bil, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = pre_bil  # reuse reference (no copy)

    hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dom_h, dom_s, dom_v = dominant_hsv_from_hist(hsv_small, min_s=MIN_S_FOR_DOMINANT, min_v=MIN_V_FOR_DOMINANT)

    hsv_full = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
    ranges = hsv_range_wrap(dom_h, dom_s, dom_v, H_TOL, S_TOL, V_TOL)

    bg_mask = inrange_multi(hsv_full, ranges)
    inverse_mask = cv2.bitwise_not(bg_mask)

    inverse_mask = cv2.dilate(inverse_mask, MORPH_KERNEL, iterations=5)
    inverse_mask = cv2.erode(inverse_mask, MORPH_KERNEL, iterations=8)

    # --- Crop to object (unchanged semantics) ---
    h_mask, w_mask = inverse_mask.shape[:2]
    non_bg_flat = np.flatnonzero(inverse_mask != 255)
    if non_bg_flat.size == 0:
        return None

    first_flat = int(non_bg_flat[0])
    last_flat = int(non_bg_flat[-1])
    crop_y0, crop_x0 = divmod(first_flat, w_mask)
    crop_y1, crop_x1 = divmod(last_flat, w_mask)

    pre_bil_cropped = pre_bil[crop_y0:crop_y1, crop_x0:crop_x1]
    inverse_mask_cropped = inverse_mask[crop_y0:crop_y1, crop_x0:crop_x1]
    if pre_bil_cropped.size == 0 or inverse_mask_cropped.size == 0:
        return None

    # --- Rectangle detection (unchanged) ---
    first_row = inverse_mask_cropped[0].astype(np.uint8)
    is_255 = (first_row == 255).astype(np.int8)
    padded = np.pad(is_255, (1, 1), mode="constant", constant_values=0)
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    rectangles = list(zip(starts.astype(int), ends.astype(int)))

    # --- Classify each band ROI ---
    mask_h2, mask_w2 = inverse_mask_cropped.shape[:2]

    # Only allocate themask/result when DEBUG_SAVE is enabled
    themask = None
    if DEBUG_SAVE and (frame_idx % DEBUG_EVERY_N == 0):
        themask = np.zeros((mask_h2, mask_w2), np.uint8)

    band_labels = []
    for start_x, end_x in rectangles:
        pad = int(abs(end_x - start_x) / 2.5)
        x_left = start_x + pad
        x_right = end_x - pad
        if x_right <= x_left:
            continue

        roi = pre_bil_cropped[0:mask_h2, x_left:x_right]
        if roi.size == 0:
            continue

        feat = band_feature_from_roi(roi)
        label, conf, score = classify_from_calibration(feat, CALIBRATION)
        band_labels.append((label, conf, score, feat["med_bgr"]))

        if themask is not None:
            themask[0:mask_h2, x_left:x_right] = 1

    if themask is not None:
        import os
        os.makedirs(DEBUG_DIR, exist_ok=True)
        result = cv2.bitwise_and(pre_bil_cropped, pre_bil_cropped, mask=themask)
        cv2.imwrite(f"{DEBUG_DIR}/result_{frame_idx:06d}.png", result)

    return {"bands": band_labels}


def main():
    cam = Picamera2()
    cam.preview_configuration.main.size = (1280, 720)
    cam.preview_configuration.main.format = "RGB888"
    cam.configure("preview")
    cam.start()
    time.sleep(1)

    meta = cam.capture_metadata()
    cam.set_controls({
        "FrameDurationLimits": (500_000, 500_000),
        "ExposureTime": 50000,
        "AnalogueGain": 1.0,
        "ColourGains": meta["ColourGains"],
        "AeEnable": False,
        "AwbEnable": False,
    })
    time.sleep(2)

    LCD1602.init(0x27, 1)

    frame_idx = 0
    next_time = time.time()

    while True:
        frame = cam.capture_array()
        frame = frame[405:585, 2:1078]  # your crop
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        out = process_frame(frame, frame_idx=frame_idx)
        if out is not None:
            bands = out["bands"]
            LCD1602.clear()
            LCD1602.write(0, 0, " ".join([lbl for (lbl, _, _, _) in bands]))
            print(" | ".join([f"{lbl}({conf:.2f})" for (lbl, conf, _, _) in bands]))

        frame_idx += 1

        # scheduler: aim for FRAME_INTERVAL_SEC, don't add fixed sleep on top
        next_time += FRAME_INTERVAL_SEC
        sleep_for = next_time - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_time = time.time()


if __name__ == "__main__":
    main()
