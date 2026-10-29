# Penjelasan `pcd_extractor.py`

Dokumen ini menjelaskan isi dan cara kerja `pcd_extractor.py` pada proyek FaciCheck. Penjelasan dibuat berdasarkan implementasi kode saat ini. Dokumen ini dapat menjadi bahan awal untuk penulisan metodologi, tetapi angka performa, klaim klinis, dan rincian eksperimen harus diisi dari hasil penelitian yang benar-benar dilakukan.

> Catatan istilah: kode tidak mendefinisikan kepanjangan “PCD”. Jangan mengembangkan singkatan tersebut di naskah sebelum kepanjangannya dikonfirmasi oleh penulis proyek.

## 1. Tujuan modul

`pcd_extractor.py` berisi kelas `PCDFeatureExtractor` untuk mengubah citra wajah menjadi sekumpulan angka yang menggambarkan geometri dan perbedaan antara sisi kanan dan kiri wajah pada citra. Modul ini menggunakan landmark dari MediaPipe Face Mesh, kemudian menghitung fitur area mata, alis, dan mulut.

Modul ini **bukan model klasifikasi dan bukan alat diagnosis**. Tugasnya adalah mendeteksi landmark dan menghitung fitur. Dalam aplikasi FaciCheck, hasilnya dipakai bersama fitur tambahan dan pemeriksaan kualitas foto sebelum fitur diberikan kepada model prediksi.

## 2. Pustaka dan metode yang digunakan

- **MediaPipe Face Mesh** mendeteksi landmark wajah. Kode mengaktifkan `refine_landmarks=True`, sehingga landmark wajah diperinci dan landmark iris juga digunakan.
- **OpenCV (`cv2`)** membaca dan memutar citra, serta mengubah format warna BGR menjadi RGB sebelum citra diproses MediaPipe.
- **NumPy** menyimpan koordinat landmark dan melakukan operasi geometri, seperti jarak Euclidean, rotasi koordinat, nilai trigonometri, dan Singular Value Decomposition (SVD).
- **Eye Aspect Ratio (EAR)** digunakan untuk mengukur rasio tinggi kelopak terhadap lebar mata.
- **Normalisasi terhadap jarak antarsudut luar mata** digunakan agar beberapa ukuran tidak terlalu bergantung pada besar-kecil wajah dalam citra.
- **SVD** digunakan untuk memperkirakan arah garis tengah wajah dari beberapa landmark.

Metode di atas menjelaskan operasi yang dilakukan program. Metode ini sendiri tidak membuktikan validitas klinis fitur atau ketepatan prediksi.

## 3. Konvensi koordinat dan sisi wajah

Landmark diubah menjadi koordinat piksel `(x, y)`. Pada citra OpenCV, `x` bertambah ke kanan dan `y` bertambah ke bawah. Nama `R` dan `L` dalam kode merujuk pada sisi anatomis subjek, bukan sisi kanan dan kiri orang yang melihat foto.

Indeks penting yang digunakan:

| Landmark | Indeks | Penggunaan |
|---|---:|---|
| Sudut luar mata kanan/kiri subjek | 33 / 263 | Arah kemiringan dan skala wajah |
| Sudut dalam mata kanan/kiri subjek | 133 / 362 | Lebar mata untuk EAR |
| Pasangan kelopak mata | 160–144, 159–145, 158–153; 385–380, 386–374, 387–373 | Tinggi kelopak mata |
| Pusat iris kanan/kiri subjek | 468 / 473 | Acuan jarak vertikal alis |
| Titik alis kanan/kiri subjek | 105 / 334 | Fitur alis |
| Sudut bibir kanan/kiri subjek | 61 / 291 | Kemiringan dan asimetri bibir |
| Garis tengah | 10, 168, 1, 152 | Perkiraan sumbu tengah wajah |

Indeks tersebut mengikuti konfigurasi landmark MediaPipe yang dipakai modul. Versi pustaka dan konfigurasi perlu dicatat agar eksperimen dapat direproduksi.

## 4. Isi dan fungsi bagian-bagian kode

### 4.1 Konstanta indeks dan `FEATURE_NAMES`

Di bagian awal file, konstanta seperti `R_EYE_OUTER`, `L_EYE_OUTER`, `R_IRIS`, dan `MIDLINE` memberi nama pada indeks landmark. Ini membuat rumus lebih mudah dibaca daripada menulis angka indeks langsung di setiap perhitungan.

`FEATURE_NAMES` mencatat delapan fitur utama yang dimaksud sebagai fitur model. `META_NAMES` mencatat empat nilai metadata untuk orientasi dan pemeriksaan kualitas. `compute_features()` mengembalikan fitur utama **beserta** metadata.

### 4.2 Konstruktor `__init__()`

Konstruktor menyiapkan objek MediaPipe Face Mesh dengan satu wajah maksimum, mode gambar statis, landmark yang diperinci, dan ambang deteksi awal yang secara bawaan bernilai `0.5`.

Parameter `auto_orient` menentukan apakah program mencoba rotasi citra lain bila orientasi awal tidak lolos pemeriksaan. Nilai bawaan adalah `True`.

### 4.3 `close()`

Menutup resource Face Mesh MediaPipe. Sebaiknya dipanggil saat proses ekstraksi selesai, terutama bila objek digunakan dalam program atau proses yang berjalan lama.

### 4.4 `_correct_head_roll()`

Metode statis ini menghitung sudut garis yang menghubungkan sudut luar mata kanan dan kiri, lalu memutar semua titik landmark di sekitar titik tengah kedua mata agar garis tersebut mendatar. Metode mengembalikan koordinat yang telah diputar dan sudut rotasi dalam radian.

Secara ringkas:

\[
\theta = \operatorname{atan2}(y_L-y_R,\;x_L-x_R)
\]

Perhitungan ini mengoreksi kemiringan di bidang gambar (roll). Koreksi ini tidak memperbaiki perubahan bentuk akibat wajah menoleh, perspektif kamera, atau pose tiga dimensi.

### 4.5 `_ear()` dan `_sym()`

`_ear()` menghitung Eye Aspect Ratio untuk satu mata. Tiga jarak vertikal kelopak dirata-ratakan, lalu dibagi dengan jarak horizontal antara sudut luar dan sudut dalam mata:

\[
EAR = \frac{\text{rata-rata tinggi kelopak}}{\text{lebar mata}+\varepsilon}
\]

`_sym()` menghitung perbedaan relatif bertanda antara dua nilai:

\[
S(L,R)=\frac{L-R}{L+R+\varepsilon}
\]

Nilai \(\varepsilon=10^{-9}\) mencegah pembagian dengan nol. Nilai yang mendekati nol berarti kedua nilai relatif mirip; tanda menunjukkan arah selisih menurut urutan argumen `L` dan `R`.

### 4.6 `compute_features()`

Metode ini menerima landmark berukuran `(478, 2)` dalam piksel dan menghasilkan kamus fitur. Urutan kerja utamanya:

1. Mengubah koordinat landmark ke tipe numerik dan mengoreksi roll.
2. Menghitung jarak antarsudut luar mata sebagai skala `d`.
3. Menghitung EAR kedua mata.
4. Menghitung jarak vertikal alis terhadap iris.
5. Menghitung kemiringan sudut bibir.
6. Memperkirakan garis tengah wajah menggunakan empat landmark dan SVD.
7. Mengukur jarak kedua sudut bibir ke garis tengah tersebut.
8. Mengembalikan fitur utama dan metadata.

Jika jarak skala tidak positif, kode menggunakan `1.0` agar pembagian tetap terdefinisi. Ini merupakan perlindungan numerik, bukan jaminan bahwa landmark yang degeneratif menghasilkan fitur yang bermakna.

### 4.7 Deteksi landmark: `_landmarks()`

Metode ini menerima satu citra BGR, mengubahnya ke RGB, lalu menjalankan MediaPipe. Jika tidak ada landmark wajah, metode mengembalikan `None`. Jika wajah ditemukan, koordinat landmark relatif MediaPipe dikalikan lebar dan tinggi citra sehingga menjadi koordinat piksel.

### 4.8 Pemeriksaan orientasi: `_is_valid_upright_face()`

Metode ini memeriksa bahwa:

- jumlah landmark cukup untuk indeks yang digunakan;
- rata-rata titik alis berada di atas mata;
- mata berada di atas mulut; dan
- jarak kedua sudut luar mata sekurang-kurangnya 10 piksel.

Urutan vertikal pada citra tegak dinyatakan sebagai:

\[
y_{\text{alis}} < y_{\text{mata}} < y_{\text{mulut}}
\]

Pemeriksaan ini adalah heuristik berbasis geometri. Pemeriksaan tersebut dapat gagal bila landmark keliru, wajah sangat terhalang, atau pose/citra berada di luar kondisi yang diharapkan.

### 4.9 `canonicalize()`

Metode ini mencoba orientasi asli terlebih dahulu. Jika tidak lolos dan `auto_orient=True`, metode mencoba citra yang diputar 180°, 90° searah jarum jam, dan 90° berlawanan arah jarum jam. Orientasi pertama yang lolos pemeriksaan dikembalikan bersama landmark dan jumlah seperempat putaran (`k`).

Bila semua percobaan gagal, metode mengembalikan citra awal, landmark `None`, dan `k=0`. Nilai `k` yang mungkin dipakai kode adalah `0`, `1`, `2`, atau `3`.

### 4.10 `extract_features()`

Ini adalah antarmuka tingkat tinggi untuk ekstraksi langsung. Input dapat berupa path file citra atau array citra BGR OpenCV. Bila path tidak dapat dibaca atau landmark tidak ditemukan setelah pemeriksaan orientasi, hasilnya `None`. Bila berhasil, metode memanggil `compute_features()` dan mengembalikan fitur beserta metadata.

## 5. Delapan fitur utama

Empat fitur pertama menyimpan besar perbedaan absolut. Empat fitur bertanda berikutnya mempertahankan arah perbedaan.

| Nama fitur | Cara hitung | Makna operasional |
|---|---|---|
| `ear_asym` | \(|EAR_L-EAR_R|\) | Besar perbedaan EAR antara mata kiri dan kanan subjek |
| `brow_asym` | \(|brow_L-brow_R|/d\) | Besar perbedaan jarak vertikal alis terhadap iris, dinormalisasi dengan jarak antarsudut luar mata |
| `mouth_tilt` | \(|y_L-y_R|/d\) pada sudut bibir | Besar selisih vertikal kedua sudut bibir yang dinormalisasi |
| `mouth_midline_asym` | \(|dist_L-dist_R|/d\) | Besar perbedaan jarak sudut bibir ke garis tengah wajah |
| `ear_asym_signed` | \(S(EAR_L,EAR_R)\) | Perbedaan EAR bertanda |
| `brow_asym_signed` | \(S(brow_L,brow_R)\) | Perbedaan ukuran alis bertanda |
| `mouth_tilt_signed` | \((y_L-y_R)/d\) | Kemiringan bibir bertanda |
| `mouth_midline_signed` | \(S(dist_L,dist_R)\) | Perbedaan jarak bibir ke garis tengah bertanda |

`d` adalah jarak antarsudut luar mata setelah koreksi roll. Istilah “asimetri” di tabel merujuk pada perbedaan geometri di citra, bukan kesimpulan klinis.

Untuk interpretasi tanda, gunakan rumus dan urutan sisi di kode. Karena koordinat `y` bertambah ke bawah, tanda positif pada fitur vertikal tidak boleh otomatis diterjemahkan sebagai sisi yang mengalami kondisi klinis tertentu tanpa validasi.

## 6. Metadata keluaran

Selain delapan fitur utama, `compute_features()` mengembalikan:

| Metadata | Arti dalam kode |
|---|---|
| `meta_roll_deg` | Sudut garis mata sebelum koreksi, dalam derajat |
| `meta_yaw_proxy` | Proksi menoleh dari perbedaan jarak ujung hidung ke kedua sudut luar mata |
| `meta_canthal_px` | Jarak antarsudut luar mata dalam piksel |
| `meta_quarter_turns` | Jumlah seperempat putaran yang digunakan oleh `canonicalize()` |

`meta_yaw_proxy` adalah proksi dua dimensi, bukan pengukuran sudut yaw tiga dimensi. Metadata dapat digunakan untuk pemeriksaan kualitas atau analisis, tetapi tidak otomatis menjadi masukan model.

## 7. Alur pemakaian di FaciCheck

Dalam backend, `main.py` membuat satu objek `PCDFeatureExtractor`. Citra yang diterima API lalu diproses oleh `foto_gate.analyze()`. Gerbang tersebut memanggil `canonicalize()`, menghitung fitur dasar dengan `compute_features()`, menghitung fitur tambahan melalui `extra_features.py`, dan memeriksa kondisi foto.

Jika foto tidak lolos gerbang, aplikasi mengembalikan penolakan beserta pelanggaran. Jika lolos, fitur dasar dan tambahan diteruskan ke model sesuai daftar kolom di `config_v4.json`. Dengan demikian:

1. `pcd_extractor.py` menghasilkan fitur geometris dasar dan metadata.
2. `extra_features.py` menghasilkan fitur tambahan.
3. `foto_gate.py` memeriksa kualitas foto dan menggabungkan fitur.
4. `main.py` memilih fitur yang dikonfigurasi dan memanggil model prediksi.

Jumlah fitur yang didefinisikan di seluruh pipeline tidak sama dengan jumlah fitur utama di `FEATURE_NAMES`. Untuk pelaporan penelitian, sebutkan secara terpisah fitur yang dihitung oleh ekstraktor dan fitur yang benar-benar digunakan sebagai masukan model.

## 8. Contoh penggunaan langsung

```python
from pcd_extractor import PCDFeatureExtractor

extractor = PCDFeatureExtractor()
try:
    features = extractor.extract_features("foto_wajah.jpg")
    if features is None:
        print("Fitur tidak dapat diekstrak dari foto.")
    else:
        print(features)
finally:
    extractor.close()
```

Contoh ini hanya menjalankan ekstraksi fitur. Ia tidak menjalankan gerbang foto, model klasifikasi, atau diagnosis.

## 9. Contoh uraian metodologi untuk jurnal

> Ekstraksi fitur geometris wajah dilakukan menggunakan MediaPipe Face Mesh dengan landmark yang diperinci. Citra diproses dalam orientasi tegak; validasi orientasi dilakukan berdasarkan susunan vertikal alis, mata, dan mulut, dengan percobaan rotasi citra bila diperlukan. Koordinat landmark dikonversi ke piksel. Kemiringan bidang gambar dikoreksi menggunakan garis yang menghubungkan kedua sudut luar mata, dan jarak antarsudut luar mata digunakan sebagai faktor normalisasi untuk fitur tertentu.
>
> Fitur utama yang dihitung meliputi perbedaan Eye Aspect Ratio kedua mata, perbedaan jarak vertikal alis terhadap iris, perbedaan vertikal sudut bibir, dan perbedaan jarak sudut bibir terhadap garis tengah wajah. Garis tengah wajah diperkirakan dari landmark dahi, pangkal hidung, ujung hidung, dan dagu menggunakan Singular Value Decomposition. Fitur dihitung dalam bentuk absolut dan, untuk beberapa ukuran, bentuk bertanda. Modul juga menghasilkan metadata kemiringan, proksi menoleh, jarak antarsudut luar mata, dan orientasi citra.

Sebelum digunakan di jurnal, tambahkan rincian yang benar-benar diketahui dari eksperimen: versi MediaPipe dan pustaka, sumber dan jumlah data, perangkat/protokol pengambilan gambar, kriteria inklusi dan eksklusi, cara memperoleh label, pembagian data, serta metode evaluasi. Jangan menyatakan nilai akurasi atau manfaat klinis tanpa hasil evaluasi yang mendukungnya.

## 10. Batasan dan hal yang perlu diperhatikan

- Akurasi fitur bergantung pada kualitas dan ketepatan landmark MediaPipe.
- Roll dikoreksi, tetapi pose menoleh/menunduk, perspektif, pencahayaan, ekspresi, dan occlusion tetap dapat memengaruhi geometri citra.
- Landmark adalah proyeksi dua dimensi; modul tidak merekonstruksi bentuk wajah tiga dimensi.
- Syarat orientasi dan jarak minimum 10 piksel adalah heuristik implementasi, bukan standar klinis.
- Perbedaan geometri tidak dengan sendirinya menunjukkan penyakit atau tingkat keparahan.
- Skrining bukan diagnosis. Validasi pada data yang sesuai dan penilaian tenaga kesehatan diperlukan sebelum membuat klaim klinis.
- Catat versi MediaPipe/OpenCV dan konfigurasi model agar proses dapat direproduksi.

