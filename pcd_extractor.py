"""
pcd_extractor.py (v3 - Fixed Auto-Orientation with Anatomical Validation)
Ekstraksi fitur asimetri wajah dari foto menggunakan MediaPipe Face Mesh.

Konvensi: "R" dan "L" = sisi anatomis SUBJEK (bukan sisi gambar).
MediaPipe: landmark 33 = sudut luar mata kanan subjek (tampak di kiri gambar),
           landmark 263 = sudut luar mata kiri subjek (tampak di kanan gambar).

Perbaikan v3:
- _quarter_turns_to_upright() dihapus karena mudah tertipu MediaPipe yang
  memaksa fit landmark pada foto terbalik (eyes_y < mouth_y tetap terpenuhi
  padahal wajah sebenarnya terbalik).
- Diganti _is_valid_upright_face() yang mengecek hirarki anatomi:
  alis (Y) < mata (Y) < mulut (Y) — kondisi yang tidak mungkin terpenuhi
  secara kebetulan pada foto terbalik.
"""
import cv2
import numpy as np
import mediapipe as mp

# --- Indeks landmark MediaPipe Face Mesh (478 titik, refine_landmarks=True) ---
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_OUTER, L_EYE_INNER = 263, 362

# Pasangan (atas, bawah) untuk tinggi kelopak; 3 pasang dirata-rata agar stabil
R_EYE_PAIRS = [(160, 144), (159, 145), (158, 153)]
L_EYE_PAIRS = [(385, 380), (386, 374), (387, 373)]

R_IRIS, L_IRIS = 468, 473          # pusat iris (hanya ada jika refine_landmarks=True)
R_BROW, L_BROW = 105, 334          # titik alis
R_MOUTH, L_MOUTH = 61, 291         # sudut bibir
MIDLINE = [10, 168, 1, 152]        # dahi, pangkal hidung, ujung hidung, dagu
NOSE_TIP = 1

EPS = 1e-9


class PCDFeatureExtractor:
    # Fitur untuk model ML (urutan tetap, dipakai sebagai kolom CSV)
    FEATURE_NAMES = [
        "ear_asym",              # |EAR_L - EAR_R|
        "brow_asym",             # asimetri jarak alis ke iris
        "mouth_tilt",            # kemiringan sudut bibir
        "mouth_midline_asym",    # asimetri jarak sudut bibir ke garis tengah
        "ear_asym_signed",       # (EAR_L - EAR_R) / (EAR_L + EAR_R)
        "brow_asym_signed",
        "mouth_tilt_signed",
        "mouth_midline_signed",
    ]
    # Metadata untuk kontrol kualitas (BUKAN fitur model)
    META_NAMES = ["meta_roll_deg", "meta_yaw_proxy", "meta_canthal_px", "meta_quarter_turns"]

    def __init__(self, min_detection_confidence=0.5, auto_orient=True):
        self.auto_orient = auto_orient
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def close(self):
        self.face_mesh.close()

    # ------------------------------------------------------------------
    # Geometri
    # ------------------------------------------------------------------
    @staticmethod
    def _correct_head_roll(pts):
        """Putar semua titik (N,2) agar garis antar sudut luar mata horizontal."""
        v = pts[L_EYE_OUTER] - pts[R_EYE_OUTER]
        theta = np.arctan2(v[1], v[0])               # sudut kemiringan garis mata
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, s], [-s, c]])            # matriks rotasi sebesar -theta
        center = (pts[L_EYE_OUTER] + pts[R_EYE_OUTER]) / 2.0
        return (pts - center) @ rot.T + center, theta

    @staticmethod
    def _ear(p, pairs, outer, inner):
        """Eye Aspect Ratio: rata-rata tinggi kelopak / lebar mata."""
        height = np.mean([np.linalg.norm(p[a] - p[b]) for a, b in pairs])
        width = np.linalg.norm(p[outer] - p[inner])
        return height / (width + EPS)

    @staticmethod
    def _sym(l, r):
        """Indeks simetri bertanda: 0 = simetris, tanda = sisi mana yang lebih besar."""
        return (l - r) / (l + r + EPS)

    def compute_features(self, landmarks_px, quarter_turns=0):
        """landmarks_px: array (478, 2) dalam piksel. Mengembalikan dict fitur + meta."""
        p, theta = self._correct_head_roll(np.asarray(landmarks_px, dtype=float))

        # Skala: jarak antar sudut luar mata (outer canthal distance)
        d = np.linalg.norm(p[L_EYE_OUTER] - p[R_EYE_OUTER])
        d = d if d > 0 else 1.0

        # 1. Mata: EAR per sisi (sudah normal terhadap lebar mata masing-masing)
        ear_r = self._ear(p, R_EYE_PAIRS, R_EYE_OUTER, R_EYE_INNER)
        ear_l = self._ear(p, L_EYE_PAIRS, L_EYE_OUTER, L_EYE_INNER)

        # 2. Alis: jarak vertikal alis -> pusat iris (setelah roll dikoreksi, vertikal = sumbu y)
        brow_r = p[R_IRIS][1] - p[R_BROW][1]
        brow_l = p[L_IRIS][1] - p[L_BROW][1]

        # 3. Bibir: kemiringan sudut bibir (y membesar = lebih rendah di gambar)
        tilt = (p[L_MOUTH][1] - p[R_MOUTH][1]) / d

        # 4. Garis tengah wajah dari beberapa titik (fit garis via SVD)
        m = p[MIDLINE]
        c = m.mean(axis=0)
        direction = np.linalg.svd(m - c)[2][0]
        normal = np.array([-direction[1], direction[0]])
        dist_r = abs((p[R_MOUTH] - c) @ normal)
        dist_l = abs((p[L_MOUTH] - c) @ normal)

        # Proxy yaw (menoleh): selisih relatif jarak ujung hidung ke kedua mata
        n_r = np.linalg.norm(p[NOSE_TIP] - p[R_EYE_OUTER])
        n_l = np.linalg.norm(p[NOSE_TIP] - p[L_EYE_OUTER])
        yaw_proxy = abs(n_l - n_r) / (n_l + n_r + EPS)

        return {
            "ear_asym": float(abs(ear_l - ear_r)),
            "brow_asym": float(abs(brow_l - brow_r) / d),
            "mouth_tilt": float(abs(tilt)),
            "mouth_midline_asym": float(abs(dist_l - dist_r) / d),
            "ear_asym_signed": float(self._sym(ear_l, ear_r)),
            "brow_asym_signed": float(self._sym(brow_l, brow_r)),
            "mouth_tilt_signed": float(tilt),
            "mouth_midline_signed": float(self._sym(dist_l, dist_r)),
            "meta_roll_deg": float(np.degrees(theta)),
            "meta_yaw_proxy": float(yaw_proxy),
            "meta_canthal_px": float(d),
            "meta_quarter_turns": int(quarter_turns),
        }

    # ------------------------------------------------------------------
    # Pipeline gambar -> fitur
    # ------------------------------------------------------------------
    def _landmarks(self, img):
        """Deteksi landmark pada satu orientasi. Return array (478, 2) piksel, atau None."""
        h, w = img.shape[:2]
        result = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        lms = result.multi_face_landmarks[0].landmark
        return np.array([[lm.x * w, lm.y * h] for lm in lms])

    @staticmethod
    def _is_valid_upright_face(pts):
        """Validasi anatomi wajah tegak:
        - Alis (Y) < Mata (Y) < Mulut (Y)   (Y lebih kecil = lebih atas di gambar)
        - Jarak antar mata cukup besar (bukan deteksi noise)
        Cek ini TIDAK bisa ditipu oleh MediaPipe yang mis-fit pada foto terbalik,
        karena pada foto terbalik alis justru berada di bawah mata.
        """
        if pts is None or len(pts) < 474:
            return False

        eyes_y  = (pts[R_EYE_OUTER][1] + pts[L_EYE_OUTER][1]) / 2.0
        mouth_y = (pts[R_MOUTH][1]     + pts[L_MOUTH][1])     / 2.0
        brow_y  = (pts[R_BROW][1]      + pts[L_BROW][1])      / 2.0

        # Urutan: alis di atas mata, mata di atas mulut
        if not (brow_y < eyes_y < mouth_y):
            return False

        # Jarak antar mata harus masuk akal (min 10px)
        canthal_px = np.linalg.norm(pts[L_EYE_OUTER] - pts[R_EYE_OUTER])
        if canthal_px < 10.0:
            return False

        return True

    _ROT = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def canonicalize(self, img):
        """Return (gambar_tegak, landmark_px, k). landmark None jika wajah tidak terdeteksi.

        Strategi (v3):
        1. Coba orientasi asli (k=0) — deteksi + validasi anatomi.
        2. Jika gagal dan auto_orient=True, coba 180°, 90°CW, 270°CW secara berurutan.
        3. Kembalikan orientasi pertama yang lolos _is_valid_upright_face().
        4. Jika semua gagal, kembalikan (img, None, 0).
        """
        # Coba orientasi asli dulu
        pts = self._landmarks(img)
        if self._is_valid_upright_face(pts):
            return img, pts, 0

        # Coba rotasi lain jika diizinkan
        if self.auto_orient:
            for k in (2, 1, 3):   # 180° dulu (paling umum), lalu 90°CW, 270°CW
                cand = cv2.rotate(img, self._ROT[k])
                cand_pts = self._landmarks(cand)
                if self._is_valid_upright_face(cand_pts):
                    return cand, cand_pts, k

        return img, None, 0

    def extract_features(self, image_input):
        """image_input: path (str) atau array BGR OpenCV. Return dict, atau None bila gagal."""
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                return None
        else:
            img = image_input
        _, pts, k = self.canonicalize(img)
        if pts is None:
            return None
        return self.compute_features(pts, quarter_turns=k)