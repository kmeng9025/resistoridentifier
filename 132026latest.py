import cv2
import numpy as np
import time
from picamera2 import Picamera2
# from LCD import LCD
import LCD1602

# =========================
# Pi/headless runtime config
# =========================
FRAME_INTERVAL_SEC = 0.5
CAMERA_INDEX = 0  # change if needed

DEBUG_SAVE = False
DEBUG_DIR = "debug_frames"  # will be created if DEBUG_SAVE=True
DEBUG_EVERY_N = 10          # save debug images every N processed frames

# =========================
# Timing config
# =========================
TIMING = True               # set False to disable timing prints
TIMING_EVERY_N = 1          # print timing every N frames (increase to reduce spam)

class StageTimer:
    """
    Lightweight stage timer. Overhead is tiny compared to OpenCV ops.
    """
    __slots__ = ("t0", "last", "times")

    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.times = []

    def mark(self, name: str):
        now = time.perf_counter()
        self.times.append((name, (now - self.last) * 1000.0))
        self.last = now

    def summary(self):
        total = (time.perf_counter() - self.t0) * 1000.0
        return total, self.times


# =========================
# Hardcoded calibration (from your calibration.json)
# =========================
# Source: uploaded calibration.json
# CALIBRATION = {
#     "classes": {
#         "brown": {
#             "chrom_mean": [0.5244386196136475, 0.30310794711112976, 0.17245343327522278],
#             "chrom_std":  [0.02690264582633972, 0.012099356390535831, 0.025169996544718742],
#             "bright_mean": 74.48719787597656,
#             "bright_std":  6.14642333984375,
#         },
#         "gold": {
#             "chrom_mean": [0.46568626165390015, 0.343137264251709, 0.19117647409439087],
#             "chrom_std":  [1e-4, 1e-4, 1e-4],
#             "bright_mean": 127.16199493408203,
#             "bright_std":  1e-4,
#         },
#         "black": {
#             "chrom_mean": [0.6581708192825317, 0.3116757273674011, 0.030153462663292885],
#             "chrom_std":  [0.014233237132430077, 0.00470907473936677, 0.011856726370751858],
#             "bright_mean": 52.66999816894531,
#             "bright_std":  2.8762624263763428,
#         },
#         "orange": {
#             "chrom_mean": [0.3285709619522095, 0.29756927490234375, 0.3738597631454468],
#             "chrom_std":  [0.0035343600902706385, 0.005319563671946526, 0.008753908798098564],
#             "bright_mean": 120.25775146484375,
#             "bright_std":  2.022850751876831,
#         },
#         "green": {
#             "chrom_mean": [0.4555160105228424, 0.5160142183303833, 0.02846975065767765],
#             "chrom_std":  [1e-4, 1e-4, 1e-4],
#             "bright_mean": 102.09900665283203,
#             "bright_std":  1e-4,
#         },
#         "red": {
#             "chrom_mean": [0.4526315927505493, 0.20701754093170166, 0.340350866317749],
#             "chrom_std":  [1e-4, 1e-4, 1e-4],
#             "bright_mean": 78.34200286865234,
#             "bright_std":  1e-4,
#         },
#         "purple": {
#             "chrom_mean": [0.8255318999290466, 0.17446808516979218, 0.0],
#             "chrom_std":  [1e-4, 1e-4, 1e-4],
#             "bright_mean": 46.18299865722656,
#             "bright_std":  1e-4,
#         },
#         "yellow": {
#             "chrom_mean": [0.4587458670139313, 0.5049505233764648, 0.03630363196134567],
#             "chrom_std":  [1e-4, 1e-4, 1e-4],
#             "bright_mean": 108.94600677490234,
#             "bright_std":  1e-4,
#         },
#     }
# }

CALIBRATION = {
    "classes": {
        "br": {
            # "n": 601,
            "chrom_mean": [
                0.4798282980918884,
                0.3002983629703522,
                0.21987439692020416
            ],
            "chrom_std": [
                0.03936861455440521,
                0.01662791520357132,
                0.048419173806905746
            ],
            "bright_mean": 66.7628402709961,
            "bright_std": 5.724446773529053
        },
        "bl": {
        # "n": 532,
            "chrom_mean": [
                0.6021039485931396,
                0.3171629011631012,
                0.08073320984840393
            ],
            "chrom_std": [
                0.0224063228815794,
                0.007957104593515396,
                0.027250956743955612
            ],
            "bright_mean": 50.54780197143555,
            "bright_std": 3.9904885292053223
        },
        "grn": {
        # "n": 55,
            "chrom_mean": [
                0.4440917372703552,
                0.5413482189178467,
                0.014560043811798096
            ],
            "chrom_std": [
                0.01433463953435421,
                0.024645671248435974,
                0.01157507486641407
            ],
            "bright_mean": 88.10596466064453,
            "bright_std": 10.14437198638916
        },
        "r": {
        # "n": 123,
            "chrom_mean": [
                0.3880232870578766,
                0.1856708973646164,
                0.42630574107170105
            ],
            "chrom_std": [
                0.015979357063770294,
                0.014997826889157295,
                0.0298507921397686
            ],
            "bright_mean": 76.12236785888672,
            "bright_std": 3.514045238494873
        },
        "pur": {
        # "n": 77,
            "chrom_mean": [
                0.7752211093902588,
                0.21175555884838104,
                0.013023421168327332
            ],
            "chrom_std": [
                0.021297158673405647,
                0.014256240800023079,
                0.009865746833384037
            ],
            "bright_mean": 49.2727165222168,
            "bright_std": 1.1102464199066162
        },
        "ylw": {
        # "n": 77,
            "chrom_mean": [
                0.41115206480026245,
                0.4983845055103302,
                0.09046328812837601
            ],
            "chrom_std": [
                0.036821361631155014,
                0.02131418138742447,
                0.03309009224176407
            ],
            "bright_mean": 105.08103942871094,
            "bright_std": 11.39084529876709
        },
        "gld": {
        # "n": 37,
            "chrom_mean": [
                0.40534672141075134,
                0.3510783612728119,
                0.24357491731643677
            ],
            "chrom_std": [
                0.02350943349301815,
                0.006786972284317017,
                0.02650662139058113
            ],
            "bright_mean": 146.63255310058594,
            "bright_std": 11.007163047790527
        },
        "org": {
        # "n": 158,
            "chrom_mean": [
                0.3136765956878662,
                0.2967498004436493,
                0.38957351446151733
            ],
            "chrom_std": [
                0.03427901491522789,
                0.00971413403749466,
                0.04143955931067467
            ],
            "bright_mean": 108.69397735595703,
            "bright_std": 10.307095527648926
        }
    }
}

# =========================
# Pipeline constants (unchanged)
# =========================
PIXELATE_SCALE = 0.18

DOWNSAMPLE_MAX_W = 320
MIN_S_FOR_DOMINANT = 90
MIN_V_FOR_DOMINANT = 40

H_TOL = 5
S_TOL = 70
V_TOL = 70


# =========================
# Feature + classification (unchanged math)
# =========================
def band_feature_from_roi(roi_bgr_uint8):
    """
    roi_bgr_uint8: HxWx3 (uint8) extracted from your band ROI
    """
    # robust median per channel
    med = np.median(roi_bgr_uint8.reshape(-1, 3), axis=0).astype(np.float32)

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


# =========================
# Background masking helpers (unchanged)
# =========================
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


# =========================
# Main per-frame processing (image ops + args unchanged)
# =========================
def process_frame(frame_bgr, frame_idx=0):
    T = StageTimer() if TIMING else None

    # --- Preprocess (unchanged ops + args) ---
    denoised = cv2.fastNlMeansDenoisingColored(frame_bgr, None, 5, 5, 7, 21)
    if T: T.mark("nlm_denoise")

    blurred = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)
    if T: T.mark("sharpen")

    img_h, img_w = sharpened.shape[:2]
    small_w = max(1, int(img_w * PIXELATE_SCALE))
    small_h = max(1, int(img_h * PIXELATE_SCALE))

    down = cv2.resize(sharpened, (small_w, small_h), interpolation=cv2.INTER_AREA)
    blocky = cv2.resize(down, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    if T: T.mark("pixelate")

    # column median snapping (vectorized, same as your column loop)
    snapped = blocky.copy()
    snapped[:, :] = np.median(blocky, axis=0)
    if T: T.mark("column_median")

    pre_bil = cv2.bilateralFilter(snapped, 10, 100, 200)
    if T: T.mark("bilateral")

    # --- Dominant background HSV selection (unchanged) ---
    h0, w0 = pre_bil.shape[:2]
    if w0 > DOWNSAMPLE_MAX_W:
        scale = DOWNSAMPLE_MAX_W / w0
        small = cv2.resize(pre_bil, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = pre_bil.copy()
    if T: T.mark("downsample_bg")

    hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dom_h, dom_s, dom_v = dominant_hsv_from_hist(
        hsv_small, min_s=MIN_S_FOR_DOMINANT, min_v=MIN_V_FOR_DOMINANT
    )
    if T: T.mark("dominant_hsv")

    hsv_full = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
    ranges = hsv_range_wrap(dom_h, dom_s, dom_v, H_TOL, S_TOL, V_TOL)

    bg_mask = inrange_multi(hsv_full, ranges)
    inverse_mask = cv2.bitwise_not(bg_mask)
    if T: T.mark("hsv_mask")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inverse_mask = cv2.dilate(inverse_mask, kernel, iterations=5)
    inverse_mask = cv2.erode(inverse_mask, kernel, iterations=8)
    if T: T.mark("morphology")

    # --- Crop to object (vectorized, same semantics) ---
    h_mask, w_mask = inverse_mask.shape[:2]
    non_bg_flat = np.flatnonzero(inverse_mask != 255)
    if non_bg_flat.size == 0:
        return None  # no object
    if T: T.mark("flatnonzero")

    first_flat = int(non_bg_flat[0])
    last_flat = int(non_bg_flat[-1])
    crop_y0, crop_x0 = divmod(first_flat, w_mask)
    crop_y1, crop_x1 = divmod(last_flat, w_mask)

    pre_bil_cropped = pre_bil[crop_y0:crop_y1, crop_x0:crop_x1]
    inverse_mask_cropped = inverse_mask[crop_y0:crop_y1, crop_x0:crop_x1]
    if T: T.mark("crop")

    if pre_bil_cropped.size == 0 or inverse_mask_cropped.size == 0:
        return None

    # --- Rectangle detection on first row (vectorized) ---
    first_row = inverse_mask_cropped[0].astype(np.uint8)
    is_255 = (first_row == 255).astype(np.int8)
    padded = np.pad(is_255, (1, 1), mode="constant", constant_values=0)
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    rectangles = [[0, int(s), 0, int(e)] for s, e in zip(starts, ends)]
    if T: T.mark("rectangles")

    # --- Fill mask + classify each band ROI ---
    mask_h = inverse_mask_cropped.shape[0]
    mask_w = inverse_mask_cropped.shape[1]
    themask = np.zeros((mask_h, mask_w), np.uint8)

    band_labels = []
    for rect in rectangles:
        pad = int(abs(rect[3] - rect[1]) / 2.5)
        x_left = rect[1] + pad
        x_right = rect[3] - pad
        if x_right <= x_left:
            continue

        themask[rect[0]:mask_h, x_left:x_right] = 1

        roi = pre_bil_cropped[rect[0]:mask_h, x_left:x_right]
        if roi.size == 0:
            continue

        feat = band_feature_from_roi(roi)
        label, conf, score = classify_from_calibration(feat, CALIBRATION)
        band_labels.append((label, conf, score, feat["med_bgr"]))
    if T: T.mark("classify")

    # --- Optional debug images (headless-safe) ---
    debug_paths = None
    if DEBUG_SAVE and (frame_idx % DEBUG_EVERY_N == 0):
        import os
        os.makedirs(DEBUG_DIR, exist_ok=True)
        result = cv2.bitwise_and(pre_bil_cropped, pre_bil_cropped, mask=themask)

        debug_paths = {
            "frame": f"{DEBUG_DIR}/frame_{frame_idx:06d}.png",
            "pre_bil": f"{DEBUG_DIR}/prebil_{frame_idx:06d}.png",
            "cropped": f"{DEBUG_DIR}/cropped_{frame_idx:06d}.png",
            "mask": f"{DEBUG_DIR}/mask_{frame_idx:06d}.png",
            "result": f"{DEBUG_DIR}/result_{frame_idx:06d}.png",
        }
        cv2.imwrite(debug_paths["frame"], frame_bgr)
        # cv2.imwrite(debug_paths["pre_bil"], pre_bil)
        # cv2.imwrite(debug_paths["cropped"], pre_bil_cropped)
        # cv2.imwrite(debug_paths["mask"], inverse_mask_cropped)
        cv2.imwrite(debug_paths["result"], result)
    if T: T.mark("debug_save")

    # --- Print timing summary ---
    if T and (frame_idx % TIMING_EVERY_N == 0):
        total_ms, stages = T.summary()
        print(f"\nFrame {frame_idx} timing:")
        for name, ms in stages:
            print(f"  {name:18s}: {ms:8.1f} ms")
        print(f"  {'TOTAL':18s}: {total_ms:8.1f} ms\n")

    return {
        "dominant_hsv": (dom_h, dom_s, dom_v),
        "rectangles": rectangles,
        "bands": band_labels,
        "debug_paths": debug_paths,
    }


# def flip(bands):
#     if ((bands[0][0] == "br" or bands[0][0] == "brown") and not (bands[-1][0] == "br" or bands[-1][0] == "brown")):
#         return 1
#     elif ((bands[-1][0] == "br" or bands[-1][0] == "brown") and not (bands[0][0] == "br" or bands[0][0] == "brown")):
#         return 0
#     return -1

# =========================
# Main loop: one frame every 0.5 seconds
# =========================
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
    frame_idx = 0
    # lcd = LCD()
    LCD1602.init(0x27, 1)   # init(slave address, background light)
    # time.sleep(2)
    while (True):
        frame = cam.capture_array()
        frame = frame[404:585, 2:1078]
        out = process_frame(frame, frame_idx=frame_idx)
        if out is not None:
            bands = out["bands"]
            # from LCD import LCD
            LCD1602.clear()
            LCD1602.write(0, 0, " ".join([f"{lbl}" for (lbl, _, _, _) in bands]))
            # LCD1602.write(1, 1, 'From SunFounder')
            # lcd.message(, 1)
            summary = " | ".join([f"{lbl}({conf:.2f})" for (lbl, conf, _, _) in bands])
            print(f"[{0}] bands: {summary}")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        time.sleep(0.5)
        frame_idx += 1


if __name__ == "__main__":
    main()
