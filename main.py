from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
import joblib
import pandas as pd

# Import modul lokal
from pcd_extractor import PCDFeatureExtractor
import foto_gate

app = FastAPI(title="FaciCheck API v4", description="API Skrining Bell's Palsy")

# 1. Wajib aktifkan CORS agar React bisa memanggil API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Memperbolehkan akses dari domain/port mana saja
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Model & Config sekali saja saat server start[cite: 19]
print("Memuat model dan konfigurasi v4...")
cfg = json.load(open("config_v4.json"))
model = joblib.load("model_v4.joblib")
extractor = PCDFeatureExtractor()
print("Model siap digunakan!")

# 3. Endpoint tes kesehatan server
@app.get("/")
def read_root():
    return {"status": "online", "message": "FaciCheck API v4 Siap Melayani"}

# 4. Endpoint Utama untuk Analisis Foto Wajah[cite: 6, 22]
@app.post("/api/predict")
async def predict_face(file: UploadFile = File(...)):
    # A. Baca file gambar yang dikirim[cite: 6, 22]
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "message": "File gambar tidak valid atau rusak."}

    # B. Cek Gerbang Foto / Quality Control via foto_gate[cite: 6]
    gate_res = foto_gate.analyze(extractor, img, mode="live")
    
    if not gate_res["ok"]:
        return {
            "ok": False,
            "status": "DITOLAK",
            "violations": gate_res["violations"],
            "messages": gate_res["messages"]
        }

    # C. Ekstraksi Fitur & Prediksi jika lolos gerbang[cite: 6, 19]
    features = gate_res["features"]
    X = pd.DataFrame([{k: features[k] for k in cfg["features"]}])
    
    # Hitung Probabilitas Bell's Palsy (0 - 1)[cite: 18, 19]
    prob_bell = float(model.predict_proba(X)[0, 1])

    # D. Tentukan Zona Keputusan berdasarkan config_v4.json[cite: 18, 19]
    if prob_bell > cfg["hi"]: # > 0.65[cite: 18, 19]
        label = "Terindikasi Bell's Palsy"
        code = "HIGH_RISK"
    elif prob_bell < cfg["lo"]: # < 0.35[cite: 18, 19]
        label = "Normal"
        code = "NORMAL"
    else: # 0.35 - 0.65[cite: 18, 19]
        label = "Kemungkinan Bell's Palsy (Gejala Ringan)"
        code = "MODERATE_RISK"

    return {
        "ok": True,
        "status": "BERHASIL",
        "landmarks": gate_res["landmarks"],
        "result": {
            "label": label,
            "code": code,
            "probability": round(prob_bell * 100, 1),
            "asymmetry_metrics": {
                "brow_asymmetry": round(features.get("brow_asym", 0), 3),
                "eye_asymmetry": round(features.get("ear_asym", 0), 3),
                "mouth_tilt": round(features.get("mouth_tilt", 0), 3)
            }
        }
    }