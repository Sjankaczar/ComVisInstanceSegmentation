# Logbook — Hari 3

**Tanggal:** 16 November 2026
**Lingkup hari ini:** fine-tune ECSeg-L 36 epoch, delapan eksperimen perbaikan, analisis per-kelas

---

## 0. Catatan tentang sumber

Hari ini kerjanya satu model saja: **ECSeg-L** (EdgeCrafter, ViT komposit 33,55 juta
parameter). Semua angka berasal dari log training dan evaluasi hari itu.

Berkas yang disimpan hanya hasil yang tidak bisa dihitung ulang: kurva per
epoch, AP per kelas, matriks konfusi, dan tabel eksperimen. Log mentah
(1 MB lebih, sebagian besar baris progress per iterasi) dibuang — isinya sudah
terwakili di berkas turunan. Config final disimpan karena tanpa itu hasilnya
tidak bisa direproduksi.

---

## 1. Hasil utama: ECSeg-L 36 epoch

Fine-tune 36 epoch di subset 24 kelas (`fix22`), fold 0, val 107 gambar
(783 anotasi). Baseline eval 640×640 stretch.

| Metrik | segm | bbox |
|---|---|---|
| mAP50-95 | **0,4395** | 0,5040 |
| mAP50 | 0,6434 | 0,6500 |
| mAP75 | 0,4509 | 0,5220 |
| AP small | 0,1631 | 0,2360 |
| AP medium | 0,3818 | 0,4320 |
| AP large | 0,6029 | 0,6700 |

Epoch terbaik **32** (mAP50-95 0,4395); epoch terakhir 35 turun tipis ke 0,4389.
Early stopping patience 10, min_delta 0,0005, min_epochs 15 — tidak terpicu.

**Gap segm vs bbox = 0,0645.** Ini menunjukkan batas kualitas topeng, bukan
deteksi: model menemukan objeknya, tetapi kontur masknya belum setepat boxnya.
Gapnya paling lebar pada AP small (0,163 vs 0,236), sesuai dugaan hari 1 bahwa
resolusi rendah di mask head adalah sumber utama kegagalan objek kecil.

### 1.1 Kurva naik-turun

| Fase | Epoch | mAP50-95 |
|---|---|---|
| Awal | 0 | 0,0016 |
| Naik cepat | 1–7 | 0,133 → 0,373 |
| Plateau | 13–27 | 0,393 → 0,433 |
| Puncak | 32 | **0,4395** |

Kurva lengkap per epoch: `reports/ecseg_l/ft36_curve.csv`.

Model mencapai 0,3323 sudah di epoch 5 — 76% dari hasil akhir dalam 6 epoch
pertama. Sisa 30 epoch hanya menambah 0,107. Ini penting untuk membaca hasil
eksperimen di §2: semua probe dijalankan 6 epoch supaya sebanding, dan pada
titik itu kurva sudah cukup landai untuk perbedaan antar-konfigurasi terlihat.

---

## 2. Delapan eksperimen perbaikan

Semua dijalankan **6 epoch** dengan baseline yang sama, acuan
mAP50-95 = **0,3323** pada epoch 5. Batas noise val (2 SD) = **±0,0205** —
perbedaan di bawah itu tidak bisa dipercaya.

### 2.1 Ringkasan

| # | Percobaan | mAP50-95 | Selisih | Putusan |
|---|---|---|---|---|
| 1 | TTA saat inferensi | — | tidak dievaluasi | dilewati |
| 2 | Aspek native 480×640 (tanpa stretch) | 0,3490 | **+0,0167** | dalam noise, tapi arah positif |
| 3 | Threshold & NMS sweep | 0,4347–0,4390 | −0,005 – 0,000 | **tidak ada yang menang** |
| 4 | Threshold per kelas (optimasi F1) | 0,3618 | **−0,0778** | jauh merugikan |
| 5 | Ukuran eval lebih besar (640×864) | 0,1170 | — | gagal konvergen |
| 6 | Kuantisasi grid mask (oracle) | 0,9587 @ 160×160 | — | plafon teoretis |
| 7 | AP per kelas | — | — | diagnosis |
| 8 | Tanpa mosaic/mixup | 0,3130 | −0,0193 | **merugikan** |

Tiga probe resolusi tambahan (native 544×736, high-res) dijalankan untuk
memisahkan pengaruh distorsi aspek dari anggaran piksel — lihat §2.4.

### 2.2 Threshold dan NMS: tidak ada yang memperbaiki

Threshold global digeser −1,0 sampai +0,5, dengan dan tanpa NMS 0,6:

| Konfigurasi | mAP50-95 | Selisih |
|---|---|---|
| thr 0,0 (baseline) | 0,4395 | — |
| thr −0,5 | 0,4390 | −0,0005 |
| thr +0,5 | 0,4362 | −0,0033 |
| thr −1,0 | 0,4363 | −0,0032 |
| thr 0,0 + NMS 0,6 | 0,4350 | −0,0045 |

Semua di bawah baseline. Threshold global 0,05/0,10/0,20 juga monoton turun
(−0,0017, −0,0075, −0,0179).

**Yang paling mencolok: threshold per kelas hasil optimasi F1 turun 0,0778.**
Threshold per kelas dioptimasi di val lalu dievaluasi di val yang sama — jadi
ini overfit murni, dan hasilnya justru lebih buruk, bukan lebih baik. Bukti
kuat bahwa kalibrasi per kelas tidak bisa dilakukan di data yang sama.

Kesimpulan: **post-processing bukan sumber perbaikan untuk model ini.**
Menaikkan skor harus datang dari model atau data, bukan dari menggeser ambang.

### 2.3 Tanpa mosaic/mixup: merugikan

Mematikan mosaic dan mixup sepenuhnya menurunkan mAP dari 0,3323 ke 0,3130
(−0,0193), nyaris menyentuh batas noise. Augmentasi agresif justru membantu,
sejalan dengan ukuran dataset yang kecil (±450 gambar latih).

### 2.4 Aspek native: satu-satunya arah positif

Dataset aslinya 4:3 (640×480), tetapi baseline melatih dan mengevaluasi di
640×640 — ada stretch vertikal. Tiga probe memisahkannya:

| Konfigurasi | Piksel relatif | mAP50-95 | Selisih |
|---|---|---|---|
| 640×640 stretch (baseline) | 1,00× | 0,3323 | — |
| **480×640 native 4:3** | 0,75× | **0,3490** | **+0,0167** |
| 544×736 native 4:3 | 0,98× | 0,3300 | −0,0023 |
| 640×864 (diperbesar) | 1,80× | 0,1170 | gagal konvergen |

Membaca ini dengan hati-hati:

- 480×640 native memberi +0,0167 meski hanya 75% piksel — arahnya benar, tetapi
  masih **di bawah batas noise ±0,0205**, jadi belum bisa disebut perbaikan pasti.
- 544×736 native (piksel hampir sama dengan baseline) hanya −0,0023 — praktis
  seri. Artinya keuntungan 480×640 bukan dari jumlah piksel, melainkan dari
  menghilangkan distorsi aspek.
- 640×864 gagal konvergen sama sekali (0,1170). Bukan bukti bahwa resolusi lebih
  besar buruk — 4 epoch tidak cukup untuk model sebesar ini dengan
  `warmup_iter: 500`.

**Keputusan:** gunakan 480×640 native untuk run berikutnya, dengan catatan bahwa
keunggulannya belum terbukti di atas noise dan perlu konfirmasi 36 epoch penuh.

### 2.5 Plafon mask: bukan penyebab utama

Eksperimen 6 memberi prediksi oracle (box dan kelas dari GT) lalu mengukur
seberapa baik topeng bisa direpresentasikan pada berbagai resolusi grid:

| Grid mask | segm mAP | AP small |
|---|---|---|
| 640×640 | 0,9897 | 0,9604 |
| 320×320 | 0,9848 | 0,9476 |
| 160×160 | 0,9587 | 0,8626 |
| 80×80 | 0,8443 | 0,5057 |

Bahkan pada grid 160×160, topeng sempurna masih menghasilkan 0,9587. Jadi
**resolusi mask head bukan hambatan** pada dataset ini — model sebenarnya
hanya mencapai 0,4395, jauh dari plafon. Penyebabnya di deteksi dan
klasifikasi, bukan di kuantisasi topeng. Ini membatalkan dugaan awal hari 1
bahwa mask head resolusi rendah adalah sumber kegagalan objek kecil; yang
benar adalah objek kecil gagal **ditemukan**, bukan gagal digambar topengnya.

---

## 3. Analisis per kelas

24 kelas, mAP 0,4395. AP lengkap di `reports/ecseg_l/perclass_ap.csv`.

| Kelompok | Kelas | AP |
|---|---|---|
| Terkuat | fire-extinguisher | 0,734 |
| | person | 0,686 |
| | box | 0,673 |
| | pan | 0,660 |
| | potted plant | 0,647 |
| | bottle | 0,629 |
| Terlemah | spoon | 0,158 |
| | light | 0,166 |
| | plate | 0,194 |
| | paper | 0,196 |
| | cabinet-merged | 0,240 |
| | toy | 0,248 |

### 3.1 Potensi perbaikan

Delapan kelas terlemah menyumbang celah terbesar:

- Kalau **kedelapan kelas itu sempurna** (AP=1): mAP 0,7002 (+0,2607)
- Kalau **masing-masing naik +0,10**: mAP 0,4729 (+0,0333)

Angka kedua lebih realistis dan sudah **di atas batas noise** (0,0333 > 0,0205).
Artinya perbaikan terarah pada 8 kelas terlemah cukup untuk menghasilkan
kenaikan yang bisa dibuktikan — tanpa perlu mengganti arsitektur.

Perhatikan pola alat makan: `spoon` 0,158, `plate` 0,194, `bowl` 0,276,
`cup` 0,357. Keempatnya objek kecil, tipis, sering berdekatan atau bertumpuk di
meja. Ini kandidat paling jelas untuk augmentasi atau resolusi lebih tinggi.

### 3.2 Matriks konfusi

Matriks lengkap: `reports/ecseg_l/confusion_matrix_counts.csv` (jumlah) dan
`reports/ecseg_l/confusion_matrix_pct.csv` (persen). Pola yang terbaca:

**Hampir semua kesalahan adalah objek yang terlewat, bukan salah kelas.**
Dari 783 anotasi val: 515 diprediksi benar, **214 terlewat** (tidak diprediksi
sama sekali), dan hanya **54 yang bocor ke kelas lain**. Sisanya 394 adalah
prediksi tanpa pasangan GT (`FALSE_POSITIVE`), dicatat di baris tersendiri.

Contoh paling jelas: GT `table-merged` (98 anotasi) diprediksi benar 67,
terlewat 28, dan hanya 3 yang bocor ke kelas lain (`box` 1, `cabinet-merged` 1,
`toy` 1). Pola yang sama pada `cabinet-merged` (5 benar, 5 terlewat, 0 bocor)
dan `bed` (3 benar, 1 terlewat, 1 ke `box`).

Proporsi terlewat tertinggi (minimal 10 GT):

| Kelas | Terlewat | Total GT | % |
|---|---|---|---|
| paper | 9 | 12 | 75% |
| light | 12 | 19 | 63% |
| spoon | 14 | 27 | 52% |
| cabinet-merged | 5 | 10 | 50% |
| book | 23 | 49 | 47% |
| stove-merged | 9 | 22 | 41% |
| toy | 19 | 49 | 39% |
| bowl | 13 | 35 | 37% |

**Implikasi:** masalahnya bukan klasifikasi, tapi recall — model tidak
mengeluarkan prediksi untuk objek-objek itu. `book` punya 23 dari 49 terlewat
padahal AP-nya 0,330, dan `toy` 19 dari 49 terlewat (AP 0,248). Ini menjelaskan
mengapa §3.1 menyebut delapan kelas terlemah: keempat kelas teratas di tabel
ini (paper, light, spoon, cabinet-merged) memang kelas dengan AP terendah, dan
penyebabnya recall, bukan kebingungan label.

Kelas bertekstur lembut seperti `blanket` (1 dari 3 benar) terlalu sedikit
sampelnya untuk disimpulkan — hanya 3 anotasi di val.

---

## 4. Keputusan hari ini

| # | Keputusan | Alasan |
|---|---|---|
| 1 | ECSeg-L dipakai sebagai hasil hari ini: 0,4395 segm, 33,5M parameter | Model terbesar yang berhasil dilatih lokal; 10,7× parameter YOLO26n-seg |
| 2 | Tidak ada post-processing yang dipakai | 11 konfigurasi threshold/NMS, semuanya ≤ baseline |
| 3 | Jangan kalibrasi threshold per kelas di val | Overfit: −0,0778 |
| 4 | Pertahankan mosaic + mixup | Mematikannya −0,0193 |
| 5 | Coba 480×640 native di run berikutnya | +0,0167, arah benar tapi belum di atas noise |
| 6 | Mask head bukan prioritas | Plafon oracle 0,9587 di grid 160×160; masalahnya di deteksi |
| 7 | Fokus perbaikan ke 8 kelas terlemah | +0,10 per kelas = +0,0333, sudah di atas noise |
| 8 | Ukuran eval >1,8× piksel butuh epoch lebih banyak | 640×864 gagal di 6 epoch |

---

## 5. Yang perlu diingat untuk hari berikutnya

- **Perbandingan antar-model belum setara.** ECSeg-L di sini dievaluasi pada
  subset 24 kelas (`fix22`) fold 0 val 107 gambar, sedangkan angka YOLO26n-seg
  hari 2 berasal dari 5-fold pada label space berbeda. Jangan menaruh kedua
  angka di satu tabel tanpa menjelaskan perbedaannya.
- **Semua probe 6 epoch.** Kesimpulan "tidak membantu" berlaku pada anggaran
  epoch itu; konfigurasi yang gagal konvergen (640×864) belum diuji adil.
- **Batas noise ±0,0205** adalah acuan untuk val 107 gambar. Semua klaim
  perbaikan harus melewati angka itu.

---

## 6. Berkas yang disimpan

| Berkas | Isi |
|---|---|
| `reports/ecseg_l/ft36_curve.csv` | 36 epoch × 6 metrik segm — kurva lengkap |
| `reports/ecseg_l/perclass_ap.csv` | AP 24 kelas, terurut |
| `reports/ecseg_l/probes.csv` | 4 konfigurasi resolusi/aspect pada epoch 6 |
| `reports/ecseg_l/threshold_sweep.csv` | 11 konfigurasi threshold/NMS |
| `reports/ecseg_l/confusion_matrix_counts.csv` | Matriks konfusi (jumlah) |
| `reports/ecseg_l/confusion_matrix_pct.csv` | Matriks konfusi (persen) |
| `reports/ecseg_l/configs/ecseg_l_ft36.yml` | Config final — prasyarat reproduksi |

Tidak disertakan: checkpoint (2× 530 MB), log mentah (1,2 MB), skrip probe
(19 berkas, sekali pakai), dan `confusion_matrix.html`.

---

## 7. Rujukan

- Keputusan model & hipotesis: `reports/LOGBOOK_HARI_1.md`
- Preprocessing, fold, dan baseline YOLO: `reports/LOGBOOK_HARI_2.md`
- Dataset: https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5
