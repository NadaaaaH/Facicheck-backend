"""
foto_gate.py - Gerbang validasi foto FASTVision (file TERPISAH; pcd_extractor.py Nada tidak diubah).
Dipakai BERSAMA oleh notebook training dan backend FastAPI supaya aturan di training = aturan di aplikasi.

    from pcd_extractor import PCDFeatureExtractor
    import foto_gate
    ext = PCDFeatureExtractor()
    hasil = foto_gate.analyze(ext, "foto.jpg", mode="live")
    hasil["ok"] / hasil["messages"] / hasil["features"] / hasil["metrics"]
"""
import numpy as np
from extra_features import _roll_correct

CFG = dict(
    face_min_frac=0.15, roll_max=20.0, yaw_max=0.15, pitch_min=0.45, pitch_max=0.80,
    ear_min=0.12, gaze_max=0.20, mouth_open_max=0.03, mouth_width_max=0.72,
    bright_min=60, bright_max=200, contrast_min=25, blur_min=30, backlight_diff=60, out_margin=0.005,
)

RULES_TRAIN = ["multi_wajah", "wajah_kecil", "keluar_bingkai", "miring", "menoleh", "dongak",
               "mata_tertutup", "lirik", "gelap_terang", "kontras", "buram"]
RULES_LIVE = RULES_TRAIN + ["backlight", "terputar", "mulut_terbuka", "senyum"]

MESSAGES = {
    "wajah_tidak_terdeteksi": "Wajah tidak terdeteksi. Pastikan seluruh wajah terlihat dan cahaya cukup.",
    "multi_wajah": "Terdeteksi lebih dari satu wajah. Pastikan hanya satu orang di foto.",
    "wajah_kecil": "Wajah terlalu jauh. Dekatkan kamera sampai wajah memenuhi bingkai.",
    "keluar_bingkai": "Wajah terpotong. Pastikan seluruh wajah ada di dalam bingkai.",
    "miring": "Kepala miring. Tegakkan kepala dan sejajarkan kamera dengan mata.",
    "menoleh": "Wajah menoleh. Hadap lurus ke kamera.",
    "dongak": "Kepala mendongak atau menunduk. Luruskan kamera setinggi mata.",
    "mata_tertutup": "Mata tertutup atau menyipit. Buka mata dengan santai.",
    "lirik": "Pandangan tidak ke kamera. Lihat lurus ke lensa.",
    "gelap_terang": "Pencahayaan terlalu gelap atau terlalu terang. Cari cahaya merata dari depan.",
    "kontras": "Wajah kurang jelas (kontras rendah). Tambah cahaya dari depan.",
    "backlight": "Cahaya dari belakang. Hadapkan wajah ke sumber cahaya, jangan membelakangi jendela atau lampu.",
    "buram": "Foto buram. Tahan kamera stabil dan bersihkan lensa.",
    "terputar": "Foto terputar atau terbalik. Ambil ulang dengan kamera tegak.",
    "mulut_terbuka": "Mulut terbuka. Tutup mulut dan rileks (wajah netral).",
    "senyum": "Terdeteksi senyum. Pasang wajah datar tanpa senyum.",
}

CONFIRMATIONS = [
    "Saya tidak memakai kacamata",
    "Alis dan dahi tidak tertutup rambut atau topi",
    "Tidak memakai masker dan tangan tidak menutupi wajah",
    "Tidak memakai filter atau beauty mode",
]

R_EYE_OUT, L_EYE_OUT = 33, 263
R_EYE_IN, L_EYE_IN = 133, 362
R_EYE_PAIRS = [(160, 144), (159, 145), (158, 153)]
L_EYE_PAIRS = [(385, 380), (386, 374), (387, 373)]
R_IRIS, L_IRIS = 468, 473
R_BROW, L_BROW = 105, 334
R_MOUTH, L_MOUTH = 61, 291
NOSE_TIP = 1
EPS = 1e-9
_FD = None


def count_faces(img_bgr):
    """Jumlah wajah 'besar' di foto (wajah kecil di latar diabaikan). Minimal 1."""
    global _FD
    import cv2
    import mediapipe as mp
    if _FD is None:
        _FD = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    res = _FD.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return 1
    areas = []
    for d in res.detections:
        b = d.location_data.relative_bounding_box
        areas.append(max(b.width, 0) * max(b.height, 0))
    big = max(areas)
    return int(sum(a >= 0.25 * big for a in areas))


def gate_metrics(img_up, pts, ft, ex, n_faces=1):
    """Ukuran-ukuran untuk gerbang. img_up = foto sudah tegak, pts = landmark (478,2)."""
    import cv2
    h, w = img_up.shape[:2]
    p = _roll_correct(np.asarray(pts, dtype=float))

    def ear(pairs, o, i):
        hh = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in pairs])
        return hh / (np.linalg.norm(pts[o] - pts[i]) + EPS)

    def gaze(iris, a, b):
        c = (pts[a] + pts[b]) / 2
        return np.linalg.norm(pts[iris] - c) / (np.linalg.norm(pts[a] - pts[b]) + EPS)

    eye_y = (p[R_EYE_OUT][1] + p[L_EYE_OUT][1]) / 2
    mouth_y = (p[R_MOUTH][1] + p[L_MOUTH][1]) / 2
    pitch = (p[NOSE_TIP][1] - eye_y) / (mouth_y - eye_y + EPS)

    x0, y0 = np.maximum(pts.min(0).astype(int), 0)
    x1, y1 = np.minimum(pts.max(0).astype(int), [w, h])
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    crop = gray[y0:y1, x0:x1]
    if crop.size < 100:
        bright = contrast = blur = bg_minus = 0.0
    else:
        bright, contrast = float(crop.mean()), float(crop.std())
        small = cv2.resize(crop, (200, max(int(200 * crop.shape[0] / crop.shape[1]), 1)))
        blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
        mx, my = int(0.2 * (x1 - x0)), int(0.2 * (y1 - y0))
        mask = np.ones(gray.shape, bool)
        mask[max(y0 - my, 0):y1 + my, max(x0 - mx, 0):x1 + mx] = False
        bg = float(gray[mask].mean()) if mask.any() else bright
        bg_minus = bg - bright

    mg = CFG["out_margin"]
    out_frame = (pts[:, 0].min() < -mg * w or pts[:, 0].max() > (1 + mg) * w or
                 pts[:, 1].min() < -mg * h or pts[:, 1].max() > (1 + mg) * h)
    return {
        "g_n_faces": int(n_faces),
        "g_face_frac": float(ft["meta_canthal_px"] / min(h, w)),
        "g_out_frame": int(out_frame),
        "g_roll": float(abs(ft["meta_roll_deg"])),
        "g_yaw": float(ft["meta_yaw_proxy"]),
        "g_pitch": float(pitch),
        "g_ear_min": float(min(ear(R_EYE_PAIRS, R_EYE_OUT, R_EYE_IN), ear(L_EYE_PAIRS, L_EYE_OUT, L_EYE_IN))),
        "g_gaze": float(max(gaze(R_IRIS, R_EYE_OUT, R_EYE_IN), gaze(L_IRIS, L_EYE_OUT, L_EYE_IN))),
        "g_bright": bright, "g_contrast": contrast, "g_bg_minus_face": float(bg_minus), "g_blur": blur,
        "g_quarter": int(ft["meta_quarter_turns"]),
        "g_mouth_open": float(ex["ctx_mouth_open"]),
        "g_mouth_width": float(ex["ctx_mouth_width"]),
    }


def violations(m, mode="live", cfg=CFG):
    """Daftar aturan yang dilanggar. mode='train' -> aturan teknis saja; mode='live' -> semua."""
    v = {
        "multi_wajah": m["g_n_faces"] > 1,
        "wajah_kecil": m["g_face_frac"] < cfg["face_min_frac"],
        "keluar_bingkai": bool(m["g_out_frame"]),
        "miring": m["g_roll"] > cfg["roll_max"],
        "menoleh": m["g_yaw"] > cfg["yaw_max"],
        "dongak": not (cfg["pitch_min"] <= m["g_pitch"] <= cfg["pitch_max"]),
        "mata_tertutup": m["g_ear_min"] < cfg["ear_min"],
        "lirik": m["g_gaze"] > cfg["gaze_max"],
        "gelap_terang": not (cfg["bright_min"] <= m["g_bright"] <= cfg["bright_max"]),
        "kontras": m["g_contrast"] < cfg["contrast_min"],
        "backlight": m["g_bg_minus_face"] > cfg["backlight_diff"],
        "buram": m["g_blur"] < cfg["blur_min"],
        "terputar": m["g_quarter"] != 0,
        "mulut_terbuka": m["g_mouth_open"] > cfg["mouth_open_max"],
        "senyum": m["g_mouth_width"] > cfg["mouth_width_max"],
    }
    rules = RULES_LIVE if mode == "live" else RULES_TRAIN
    return [r for r in rules if v[r]]


def analyze(ext, image_input, mode="live", cfg=CFG):
    """Satu pintu: foto -> {ok, violations, messages, features, metrics}. ext = PCDFeatureExtractor."""
    import cv2
    from extra_features import extra_features
    img = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    if img is None:
        return {"ok": False, "violations": ["wajah_tidak_terdeteksi"],
                "messages": [MESSAGES["wajah_tidak_terdeteksi"]], "features": None, "metrics": None}
    img_up, pts, k = ext.canonicalize(img)
    if pts is None:
        return {"ok": False, "violations": ["wajah_tidak_terdeteksi"],
                "messages": [MESSAGES["wajah_tidak_terdeteksi"]], "features": None, "metrics": None}
    ft = ext.compute_features(pts, quarter_turns=k)
    ex = extra_features(pts)
    m = gate_metrics(img_up, pts, ft, ex, count_faces(img_up))
    viol = violations(m, mode, cfg)
    landmarks = {
        "brow_left": [float(value) for value in pts[L_BROW]],
        "brow_right": [float(value) for value in pts[R_BROW]],
        "eye_left": [float(value) for value in pts[L_IRIS]],
        "eye_right": [float(value) for value in pts[R_IRIS]],
        "mouth_left": [float(value) for value in pts[L_MOUTH]],
        "mouth_right": [float(value) for value in pts[R_MOUTH]],
    }
    return {"ok": not viol, "violations": viol, "messages": [MESSAGES[r] for r in viol],
            "features": {**ft, **ex}, "metrics": m, "landmarks": landmarks}
