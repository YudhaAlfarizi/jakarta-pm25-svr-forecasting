# Peramalan Nilai PM2.5 Harian DKI Jakarta Menggunakan Support Vector Regression dengan Time Series Cross-Validation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVR-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Deskripsi

Proyek ini membangun model **Support Vector Regression (SVR)** berkernel **RBF** untuk meramalkan **nilai PM2.5 harian DKI Jakarta satu hari ke depan** (*one-step-ahead forecasting*), menggunakan data Indeks Standar Pencemar Udara (ISPU) periode 2021–2025. Model dievaluasi dengan validasi yang menjaga urutan waktu (*time series cross-validation*), tanpa kebocoran data, dan dibandingkan dengan baseline *naive (persistence)*.

> **Hasil utama:** Model SVR (fitur hasil forward selection: `pm25_lag1` dan `pm25_roll14`) mencapai RMSE 16,61 pada data uji, dibandingkan RMSE 18,65 untuk baseline naive — perbaikan sebesar 10,9%. R² = 0,53, artinya model menjelaskan sekitar separuh variasi nilai PM2.5 harian.

## Notebook

| # | Notebook | Isi | Colab |
|---|---|---|---|
| 1 | [`01_data_wrangling`](01_data_wrangling.ipynb) | Membaca data, memilih periode, memeriksa dan menangani missing value | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YudhaAlfarizi/jakarta-pm25-svr-forecasting/blob/main/01_data_wrangling.ipynb) |
| 2 | [`02_eda`](02_eda.ipynb) | Statistik deskriptif, tren, pola bulanan, distribusi, ACF/PACF | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YudhaAlfarizi/jakarta-pm25-svr-forecasting/blob/main/02_eda.ipynb) |
| 3 | [`03_feature_engineering`](03_feature_engineering.ipynb) | Fitur lag, rolling mean, fitur musiman, split kronologis latih/uji | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YudhaAlfarizi/jakarta-pm25-svr-forecasting/blob/main/03_feature_engineering.ipynb) |
| 4 | [`04_svr_modeling`](04_svr_modeling.ipynb) | Baseline naive, **forward selection** (memilih fitur final dari kandidat lag notebook 02-03 berdasarkan performa), tuning SVR dengan `TimeSeriesSplit`, evaluasi, analisis residual, dan eksperimen variabel polutan tambahan | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YudhaAlfarizi/jakarta-pm25-svr-forecasting/blob/main/04_svr_modeling.ipynb) |

## Dokumentasi

- [`docs/dataset.md`](docs/dataset.md) — sumber data, kolom, karakteristik dan masalah kualitas data, cara mengunduh
- [`docs/metodologi.md`](docs/metodologi.md) — alur penelitian, alasan tiap langkah, landasan matematis SVR, referensi

## Tujuan Penelitian

1. Memahami karakteristik dan pola nilai PM2.5 harian DKI Jakarta.
2. Membangun model SVR berbasis *lag features* untuk peramalan satu hari ke depan.
3. Menentukan hyperparameter terbaik (kernel, *C*, *ε*, *γ*) dengan `TimeSeriesSplit`.
4. Mengevaluasi model (RMSE, MAE, sMAPE, R²) terhadap baseline naive.
5. Menganalisis residual untuk melihat pola kesalahan model.
6. Menguji secara empiris apakah variabel polutan lain (PM10, CO, O₃, dll.) menambah akurasi.

## Ringkasan Data

- **Sumber:** Indeks Standar Pencemar Udara (ISPU) DKI Jakarta, diterbitkan Dinas Lingkungan Hidup Provinsi DKI Jakarta melalui portal Satu Data Indonesia
- **File:** `ispu_dki_all.csv`, gabungan data 5 stasiun pemantau kualitas udara (SPKU)
- **Periode dipakai:** 1 Januari 2021 – 28 Februari 2025 (nilai PM2.5 baru tersedia sejak 2021)
- **Target:** `pm25`, yaitu nilai ISPU PM2.5 tertinggi di antara 5 stasiun pada hari tersebut
- **Satuan:** ISPU adalah **angka indeks tanpa satuan** (dikonfirmasi Dinas Lingkungan Hidup DKI Jakarta), bukan konsentrasi µg/m³

Detail lengkap dan angka yang dihasilkan notebook 01: [`docs/dataset.md`](docs/dataset.md).

## Hasil

| Model | RMSE | MAE | sMAPE (%) | R² |
|---|---|---|---|---|
| Naive (persistence) | 18.65 | 14.00 | 19.73 | 0.41 |
| **SVR (fitur terpilih, tuned)** | 16.61 | 12.93 | 17.85 | 0.53 |
| SVR + CO, O₃ | 17.00 | 13.27 | 18.48 | 0.51 |
| SVR + PM10 | 18.87 | 14.45 | 19.81 | 0.37 |
| SVR + semua polutan | 20.29 | 15.88 | 21.82 | 0.28 |


![Aktual vs Prediksi](docs/actual_vs_pred.png)

**Interpretasi:** SVR mengungguli baseline naive sebesar ±11% pada RMSE. Forward selection ([`docs/forward_selection.png`](docs/forward_selection.png)) memilih fitur yang jauh lebih sederhana dari dugaan awal — hanya `pm25_lag1` dan `pm25_roll14` (rata-rata 14 hari) yang terbukti membantu performa; lag-lag lain yang tampak signifikan di PACF (mis. lag 7, 14, 21 sebagai lag tunggal) ternyata tidak menambah performa model. Penambahan CO dan O₃ sedikit membantu, sementara PM10 dan kombinasi semua polutan justru memperburuk hasil, kemungkinan karena berkurangnya data latih akibat celah panjang pada PM10 (lihat `docs/dataset.md`). Analisis residual menunjukkan model kesulitan mengikuti lonjakan ekstrem (mis. akhir November 2024, residual sampai ±55 poin), konsisten dengan sifat kernel RBF yang menarik prediksi kembali ke rata-rata saat nilai jauh di luar rentang data latih.

Tabel prediksi harian lengkap (aktual vs SVR vs naive, 210 hari data uji): [`docs/hasil_prediksi.csv`](docs/hasil_prediksi.csv).

## Keterbatasan

- Nilai PM2.5 hanya tersedia ±4 tahun, sehingga siklus musiman teramati terbatas.
- Target adalah nilai tertinggi dari 5 stasiun pada hari itu, bukan pengukuran satu lokasi.
- ISPU adalah angka indeks, bukan konsentrasi fisik, sehingga tidak bisa dibandingkan langsung dengan standar µg/m³ negara lain.
- Periode uji (akhir 2024–awal 2025) cenderung lebih rendah nilainya dibanding data latih.
- Model hanya meramalkan satu langkah ke depan dan tidak memakai variabel meteorologi.

## Struktur Repositori

```
jakarta-pm25-svr-forecasting/
├── data/
│   ├── ispu_dki_all.csv            # data mentah
│   ├── pm25_daily.csv              # hasil notebook 01
│   ├── train.csv                   # hasil notebook 03
│   └── test.csv                    # hasil notebook 03
├── docs/
│   ├── dataset.md
│   ├── metodologi.md
│   ├── hasil_prediksi.csv          # tabel prediksi harian, hasil notebook 04
│   ├── pm25_sebelum_cleaning.png   # hasil notebook 01
│   ├── pm25_sesudah_cleaning.png   # hasil notebook 01
│   ├── pm25_distribusi.png         # hasil notebook 02
│   ├── pm25_trend.png              # hasil notebook 02
│   ├── pm25_pola_bulanan.png       # hasil notebook 02
│   ├── pm25_stl_decomposition.png  # hasil notebook 02
│   ├── pm25_acf_pacf.png           # hasil notebook 02
│   ├── korelasi_polutan.png        # hasil notebook 02
│   ├── forward_selection.png       # hasil notebook 04
│   ├── actual_vs_pred.png          # hasil notebook 04
│   └── residuals.png               # hasil notebook 04
├── 01_data_wrangling.ipynb
├── 02_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_svr_modeling.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

## Cara Menjalankan

**Google Colab (disarankan):** klik tombol Colab di tabel notebook, lalu *Runtime → Run all*, dimulai dari notebook 1.

**Lokal:**

```bash
git clone https://github.com/USERNAME/jakarta-pm25-svr-forecasting.git
cd jakarta-pm25-svr-forecasting
pip install -r requirements.txt
jupyter notebook
```

## Penulis

**[Yudha Alfarizi]** — Lulusan Matematika S1 · [LinkedIn]([#](https://www.linkedin.com/in/yudha-alfarizi-07588a286/)) · [Email]([#](https://mail.google.com/mail/?view=cm&fs=1&to=yudhaaakerja29072026@gmail.com))

## Lisensi

Kode dirilis di bawah lisensi MIT. Dataset mengikuti lisensi sumber aslinya (Kaggle).
