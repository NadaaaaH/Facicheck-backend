"""
extra_features.py (v2) - fitur tambahan di atas 8 fitur pcd_extractor.py (Nada). TIDAK mengubah pcd_extractor.py.
Input: landmark (478, 2) piksel yang sudah ditegakkan oleh PCDFeatureExtractor.canonicalize(). Hanya butuh numpy.
"R"/"L" = sisi anatomis SUBJEK (indeks kecil = kanan subjek, tampak di kiri gambar).
Kelompok: named (asimetri), context (ekspresi simetris), pairs (40 pasang landmark).
Tambahan v2: extra_features(pts) = named + context (dipakai foto_gate dan model), tanpa pairs.
"""
import numpy as np

EPS = 1e-9
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_OUTER, L_EYE_INNER = 263, 362
MIDLINE = [10, 168, 1, 152]
BROW_PTS = {"outer": (70, 300), "mid": (105, 334), "inner": (107, 336)}
MOUTH_CORNER = (61, 291)
LIP_UP_C, LIP_LO_C = 13, 14
ALA = (129, 358)
CHEEK = (205, 425)
FACE_EDGE = (234, 454)
JAW_LOW = (58, 288)
JAW_CHIN = (172, 397)
UPPER_LIP_ARC = (37, 267)
LOWER_LIP_ARC = (84, 314)
EYE_TOP_BOT = [((159, 145), (386, 374)), ((160, 144), (387, 373)), ((158, 153), (385, 380))]
MIRROR_PAIRS = [
    (33, 263), (133, 362), (159, 386), (145, 374), (160, 387), (144, 373), (158, 385), (153, 380),
    (70, 300), (63, 293), (105, 334), (66, 296), (107, 336), (46, 276), (53, 283), (52, 282),
    (65, 295), (55, 285), (61, 291), (146, 375), (91, 321), (181, 405), (84, 314), (37, 267),
    (39, 269), (40, 270), (129, 358), (98, 327), (205, 425), (50, 280), (117, 346), (123, 352),
    (234, 454), (93, 323), (132, 361), (58, 288), (172, 397), (136, 365), (150, 379), (149, 378),
]


def _roll_correct(pts):
    """Sama persis dengan PCDFeatureExtractor._correct_head_roll."""
    v = pts[L_EYE_OUTER] - pts[R_EYE_OUTER]
    theta = np.arctan2(v[1], v[0])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, s], [-s, c]])
    center = (pts[L_EYE_OUTER] + pts[R_EYE_OUTER]) / 2.0
    return (pts - center) @ rot.T + center


def _geometry(landmarks_px):
    p = _roll_correct(np.asarray(landmarks_px, dtype=float))
    d = np.linalg.norm(p[L_EYE_OUTER] - p[R_EYE_OUTER])
    d = d if d > 0 else 1.0
    m = p[MIDLINE]
    c = m.mean(axis=0)
    direction = np.linalg.svd(m - c)[2][0]
    normal = np.array([-direction[1], direction[0]])
    return p, d, c, normal


def compute_all(landmarks_px):
    """Return (named, context, pairs): tiga dict berurutan."""
    p, d, c, normal = _geometry(landmarks_px)
    dist = lambda i: abs((p[i] - c) @ normal)
    y = lambda i: p[i][1]
    eye_y_r = (y(R_EYE_OUTER) + y(R_EYE_INNER)) / 2.0
    eye_y_l = (y(L_EYE_OUTER) + y(L_EYE_INNER)) / 2.0
    lip_c_y = (y(LIP_UP_C) + y(LIP_LO_C)) / 2.0
    named = {}

    def add(name, l, r, ratio):
        s = (l - r) / (l + r + EPS) if ratio else (l - r)
        named[f"x_{name}_signed"] = float(s)
        named[f"x_{name}_abs"] = float(abs(s))

    for nm, (ri, li) in BROW_PTS.items():
        add(f"brow_{nm}", (eye_y_l - y(li)) / d, (eye_y_r - y(ri)) / d, True)
    ri, li = MOUTH_CORNER
    add("mouth_below_eye", (y(li) - eye_y_l) / d, (y(ri) - eye_y_r) / d, True)
    add("mouth_vs_lipcenter", (y(li) - lip_c_y) / d, (y(ri) - lip_c_y) / d, False)
    add("upperlip_arc", (y(UPPER_LIP_ARC[1]) - lip_c_y) / d, (y(UPPER_LIP_ARC[0]) - lip_c_y) / d, False)
    add("lowerlip_arc", (y(LOWER_LIP_ARC[1]) - lip_c_y) / d, (y(LOWER_LIP_ARC[0]) - lip_c_y) / d, False)
    add("ala_dist", dist(ALA[1]), dist(ALA[0]), True)
    add("ala_height", (y(ALA[1]) - eye_y_l) / d, (y(ALA[0]) - eye_y_r) / d, False)
    add("cheek_height", (y(CHEEK[1]) - eye_y_l) / d, (y(CHEEK[0]) - eye_y_r) / d, True)
    add("cheek_dist", dist(CHEEK[1]), dist(CHEEK[0]), True)
    add("nasolabial_len",
        np.linalg.norm(p[ALA[1]] - p[MOUTH_CORNER[1]]) / d,
        np.linalg.norm(p[ALA[0]] - p[MOUTH_CORNER[0]]) / d, True)
    h_r = np.mean([np.linalg.norm(p[a] - p[b]) for (a, b), _ in EYE_TOP_BOT]) / d
    h_l = np.mean([np.linalg.norm(p[a] - p[b]) for _, (a, b) in EYE_TOP_BOT]) / d
    add("eye_open", h_l, h_r, True)
    add("face_half", dist(FACE_EDGE[1]), dist(FACE_EDGE[0]), True)
    add("jaw_low", dist(JAW_LOW[1]), dist(JAW_LOW[0]), True)
    add("jaw_chin", dist(JAW_CHIN[1]), dist(JAW_CHIN[0]), True)

    ctx = {}
    ctx["ctx_mouth_width"] = np.linalg.norm(p[MOUTH_CORNER[0]] - p[MOUTH_CORNER[1]]) / d
    ctx["ctx_mouth_open"] = np.linalg.norm(p[LIP_UP_C] - p[LIP_LO_C]) / d
    ctx["ctx_smile_lift"] = ((lip_c_y - y(MOUTH_CORNER[0])) + (lip_c_y - y(MOUTH_CORNER[1]))) / 2.0 / d
    ctx["ctx_eye_open"] = (h_r + h_l) / 2.0
    ctx["ctx_brow_height"] = np.mean([((eye_y_l - y(li)) + (eye_y_r - y(ri))) / 2.0 / d
                                      for ri, li in BROW_PTS.values()])
    face_w = np.linalg.norm(p[FACE_EDGE[0]] - p[FACE_EDGE[1]])
    ctx["ctx_mouth_to_face"] = ctx["ctx_mouth_width"] * d / (face_w + EPS)
    ctx["ctx_nose_width"] = np.linalg.norm(p[ALA[0]] - p[ALA[1]]) / d
    ctx = {k: float(v) for k, v in ctx.items()}

    pairs = {}
    for r, l in MIRROR_PAIRS:
        pairs[f"pair_dy_{r}_{l}"] = float((y(l) - y(r)) / d)
        pairs[f"pair_dm_{r}_{l}"] = float((dist(l) - dist(r)) / d)
    return named, ctx, pairs


def _probe_names():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 200, size=(478, 2))
    pts[R_EYE_OUTER] = [50, 100]; pts[L_EYE_OUTER] = [150, 100]
    n, c, pr = compute_all(pts)
    return list(n), list(c), list(pr)


NAMED_NAMES, CONTEXT_NAMES, PAIR_NAMES = _probe_names()
FEATURE_GROUPS = {"named": NAMED_NAMES, "context": CONTEXT_NAMES, "pairs": PAIR_NAMES}


def compute_extra_features(landmarks_px, groups=("named", "context", "pairs")):
    named, ctx, pairs = compute_all(landmarks_px)
    allf = {"named": named, "context": ctx, "pairs": pairs}
    out = {}
    for g in groups:
        out.update(allf[g])
    return out


def extra_features(landmarks_px):
    """named + context (tanpa pairs). Dipakai foto_gate.analyze dan model produksi."""
    return compute_extra_features(landmarks_px, groups=("named", "context"))
