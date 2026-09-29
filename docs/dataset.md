# Dataset

[← Kembali ke README](../README.md)

## 1. Sumber Data

- **Nama:** Indeks Standar Pencemar Udara (ISPU) DKI Jakarta
- **Sumber:** Kaggle
- **Penerbit:** Dinas Lingkungan Hidup Provinsi DKI Jakarta
- **Portal:** Satu Data Indonesia (`katalog.data.go.id`), dipublikasikan per tahun (2010–2023) dengan kolom yang sama seperti `ispu_dki_all.csv`: `tanggal, stasiun, pm10, so2, co, o3, no2, max, critical, categori`
- **File yang dipakai:** `ispu_dki_all.csv`
- **Cakupan file:** 1 Januari 2010 – 28 Februari 2025

[https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021]


## 2. Kolom

| Kolom | Keterangan | Dipakai di model? |
|---|---|---|
| `tanggal` | Tanggal pengukuran | Ya (indeks waktu) |
| `stasiun` | Stasiun dengan nilai ISPU tertinggi pada hari itu | Tidak |
| `pm25` | Nilai ISPU PM2.5 | **Ya (target)** |
| `pm10`, `so2`, `co`, `o3`, `no2` | Nilai ISPU polutan lain | Hanya di eksperimen (notebook 04) |
| `max` | Nilai tertinggi di antara semua polutan | Tidak (kebocoran data) |
| `critical` | Polutan dengan nilai tertinggi | Tidak (kebocoran data) |
| `categori` | Kategori kualitas udara | Tidak (kebocoran data) |

`max`, `critical`, dan `categori` diturunkan langsung dari polutan pada hari yang sama, sehingga tidak boleh dipakai untuk meramalkan `pm25` hari itu.

## 3. Satuan: ISPU adalah angka indeks tanpa satuan

ISPU **bukan** konsentrasi fisik (µg/m³), melainkan angka indeks hasil konversi dari konsentrasi polutan, dipakai untuk menggambarkan mutu udara ambien berdasarkan dampaknya terhadap kesehatan. Ini dinyatakan langsung oleh Kepala Dinas Lingkungan Hidup DKI Jakarta dan diatur dalam **Peraturan Menteri LHK No. 14 Tahun 2020 tentang Indeks Standar Pencemar Udara**.

Konsekuensinya:
- Jangan menulis "konsentrasi PM2.5 (µg/m³)" di laporan; gunakan istilah **"nilai PM2.5"** atau **"indeks PM2.5"**.
- Nilai tidak bisa dibandingkan langsung dengan ambang WHO atau EPA yang memakai µg/m³.
- Skala indeks berkorespondensi dengan kategori kualitas udara (mis. Baik, Sedang, Tidak Sehat), sesuai Permen LHK No. 14/2020.

## 4. Periode yang dipakai: 2021–2025
Periode yang dipakai untuk pemodelan adalah **1 Januari 2021 – 28 Februari 2025**.

Jumlah hari (kalender): 1,520.
Tanggal hilang        : 0.
pm25 NaN sebelum      : 4.
pm25 NaN sesudah      : 0
Batas interpolasi     : 3 hari

pm25  :    4 hari NaN dalam   4 celah (0 celah > 3 hari)
pm10  :  200 hari NaN dalam  30 celah (10 celah > 3 hari)
so2   :   22 hari NaN dalam  16 celah (0 celah > 3 hari)
co    :   22 hari NaN dalam  14 celah (0 celah > 3 hari)
o3    :   23 hari NaN dalam  15 celah (1 celah > 3 hari)
no2   :   14 hari NaN dalam  11 celah (0 celah > 3 hari)

## 5. `ispu_dki_all.csv` adalah gabungan 5 stasiun

Satu baris per tanggal. Kolom `stasiun` menunjukkan **stasiun mana yang nilainya tertinggi pada hari itu**, dan berganti-ganti setiap hari. Seri yang dimodelkan adalah:

> **nilai ISPU PM2.5 tertinggi di antara 5 stasiun pemantau DKI Jakarta per hari.**

Implikasi:
- `stasiun` **tidak dipakai sebagai fitur**, karena tidak diketahui sebelum hari itu terjadi.
- Data **tidak dipisah per stasiun** dari file ini, karena tiap stasiun hanya muncul di sebagian hari (perlu diverifikasi ulang pola kemunculannya di notebook 01) sehingga seri per stasiun menjadi bolong.
- Ada kemungkinan bias seleksi: stasiun yang kebetulan tertinggi hari itu yang tercatat, bukan representasi rata-rata kota.
- Untuk analisis per stasiun, dataset per tahun di Satu Data Indonesia atau file `ispu_dki 1–5` bisa dipakai sebagai pengembangan lanjutan.

## 6. Penanganan missing value

Aturan yang dipakai di notebook 01 dan 03:

1. Deret disusun ke frekuensi harian (`asfreq('D')`) agar tanggal yang hilang ikut terlihat sebagai baris kosong.
2. Celah **≤ 3 hari** diisi dengan interpolasi berbasis waktu:
   ```python
   df['pm25'] = df['pm25'].interpolate(method='time', limit=3, limit_area='inside')
   ```
3. Celah **lebih panjang tidak diisi**. Baris yang fitur lag/rolling-nya melewati celah tersebut dibuang saat feature engineering.

Batas 3 hari adalah **keputusan desain**, bukan aturan baku, dipilih agar interpolasi hanya menyentuh celah yang jauh lebih pendek daripada dinamika harian PM2.5. Perlu dicatat juga bahwa interpolasi memakai nilai di kedua sisi celah, sehingga sedikit informasi "masa depan" ikut masuk ke titik yang diisi — dampaknya kecil karena hanya celah pendek yang diisi, tapi tetap dicatat sebagai keterbatasan (lihat [`metodologi.md`](metodologi.md)).

Variabel pendukung seperti `pm10` diketahui memiliki celah yang jauh lebih panjang pada 2023, dan celah itu **tidak diinterpolasi** karena akan menciptakan data buatan yang tidak realistis di musim puncak. Skenario model yang memakai `pm10` karena itu memakai data latih yang lebih sedikit dibanding skenario yang tidak memakainya. Angka pasti panjang celah ini divalidasi ulang di notebook 01.

## 7. Bentuk data di tiap tahap

**Data mentah**

```
tanggal     stasiun              pm25  pm10  so2  co    o3    no2   max   critical  categori
2010-01-01  DKI1 (Bunderan HI)   NaN   60.0  4.0  73.0  27.0  14.0  73.0  CO        SEDANG
```

**Setelah notebook 01** (`data/pm25_daily.csv`, periode 2021–2025 saja)

```
tanggal     pm25  pm10  so2   co    o3    no2
2021-01-01  58.0  38.0  2.0   11.0  65.0  6.0
```

**Setelah notebook 03** (`data/train.csv`, `data/test.csv`)

```
tanggal     pm25  pm25_lag1  pm25_lag2  pm25_lag3  pm25_lag7  pm25_roll7  month_sin  month_cos
2021-01-08  55.0  50.0       60.0       89.0       58.0       69.29       0.5        0.87
```

Baris pertama mulai 7 hari setelah data tersedia karena `lag7` membutuhkan riwayat 7 hari sebelumnya.

## 8. Variabel yang diuji sebagai eksperimen (notebook 04)

Model utama hanya memakai riwayat `pm25` sendiri. Sebagai eksperimen pembanding, diuji penambahan:

- `co`, `o3` (korelasi sedang dengan `pm25`)
- `pm10` (korelasi tertinggi, tetapi data latihnya berkurang karena celah panjang 2023)
- seluruh polutan sekaligus (`pm10`, `so2`, `co`, `o3`, `no2`)

Semua variabel pendukung memakai nilai **hari sebelumnya** (`shift(1)`), karena pada saat meramalkan hari t, nilai polutan pada hari t belum diketahui.

## 9. Keterbatasan dataset

1. Nilai PM2.5 hanya tersedia ±4 tahun dari keseluruhan rentang file (2010–2025).
2. File gabungan bukan seri satu stasiun, sehingga interpretasi spasial terbatas.
3. ISPU adalah angka indeks, bukan konsentrasi fisik.
4. `pm10` memiliki celah panjang di 2023 yang membatasi skenario eksperimen yang memakainya.
5. Tidak ada variabel meteorologi dalam file ini.

## 10. Cara mendapatkan data

1. Unduh `ispu_dki_all.csv` dari sumber pada bagian [Sumber Data](#1-sumber-data).
2. Simpan sebagai `data/ispu_dki_all.csv`.
3. Jalankan `01_data_wrangling.ipynb`, yang menghasilkan `data/pm25_daily.csv`.
