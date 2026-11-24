# Logbook — Hari 2

**Tanggal:** 15 November 2026
**Lingkup hari ini:** preprocessing, pembuatan fold, pilot Modal.com, training penuh, peningkatan skor

---

## 0. Catatan tentang sumber

Seluruh angka di sini berasal dari laporan kerja hari itu (preprocessing, pilot,
full training, peningkatan skor). Laporan-laporan aslinya sudah digabung ke
dokumen ini lalu dihapus agar tidak ada duplikasi.

Angka yang bisa dihitung ulang, dihitung ulang dari berkas yang tersimpan:
tabel per fold di §4.1 berasal dari `artifacts/csv_*.csv`, bukan disalin dari
laporan lama. Angka yang tidak bisa dihitung ulang (waktu, biaya, mAP pilot)
disalin dari laporan hari itu.

Yang disimpan sebagai berkas terpisah hanya yang tidak bisa diringkas ke teks:
konfigurasi final, hasil per run, dan skrip yang menghasilkannya (lihat §7).

---

## 1. Preprocessing

### 1.1 Hasil akhir

| | Sebelum | Sesudah | Selisih |
|---|---|---|---|
| Gambar | 922 | **906** | −16 |
| Anotasi | 11.845 | **11.265** | −580 |
| Kategori | 75 | **70** | −5 |
| Koordinat keluar batas | 1.891 anotasi | **0** | — |
| Anotasi duplikat | 31 | **0** | — |
| File hilang / id duplikat / anotasi orphan | 0 | **0** | — |

Distribusi: train 387 gambar / 4.091 anotasi · valid 160 / 1.857 · test
**359 / 5.317 (utuh, tidak disentuh)**. Things 52 kelas (6.893 anotasi),
stuff 18 kelas (4.372).

Pengurangan 16 gambar = 3 gambar kebocoran (§1.4) + 13 gambar yang kehilangan
seluruh anotasinya setelah kelas dibuang (§1.3).

### 1.2 Keputusan kategori

**Dibuang (3):**

| Kelas | Alasan |
|---|---|
| `indoor-objects` | 0 anotasi di seluruh dataset. Bukan kelas objek, melainkan root `supercategory` yang diekspor Roboflow sebagai kategori |
| `wallet` | Foto produk studio (terverifikasi visual di EDA), 197 train vs 5 test |
| `trash-bin` | Foto produk studio, 153 train vs 4 test |

**Digabung (2):** `scissor` (2 anotasi) → `scissors` (6); `st-` (1) → `stairs`
(175 train / 67 valid / 6 test). Yang kedua diverifikasi lewat render mask: mask
menutupi tangga lengkap dengan railing pada `train/182169356`, dan kelas
`stairs` sudah ada tanpa anotasi di gambar itu.

**Dipertahankan:** `pan` (225/34, sebagian besar adegan dapur sah) dan `toy`
(217/41, konteks kamar).

**Flag stuff/things:** 18 kelas `isthing=0` (`wall-*`, `floor-*`,
`ceiling-merged`, `rug-merged`, `door-stuff`, `mirror-stuff`, `window-*`,
`curtain`, `stairs`, `tree-merged`, `counter`), 52 `isthing=1`. Ditulis ke
`categories.json` dan ke setiap file COCO, sehingga AP things-only bisa dihitung
tanpa filter nama.

### 1.3 Gambar yang dibuang

13 gambar kehilangan seluruh anotasinya lalu dibuang: 11 hanya berisi
`trash-bin` (7 train, 4 valid), 2 hanya `wallet` (valid), 1
(`valid/182169175`) memang sudah tanpa anotasi sejak awal.

### 1.4 Perbaikan gambar transposed

Empat gambar punya JSON 640×480 tetapi file 480×640. Arah rotasi **ditentukan
dari data**, bukan ditebak, lewat tiga bukti independen: skor tepi-Canny pada
pita batas mask, verifikasi visual per-anotasi, dan keterbacaan teks.

| Gambar | Arah | Bukti |
|---|---|---|
| `train/182169121` | rot90 CW | skor band 5.203 vs 1.875 |
| `valid/182169109` | rot90 **CCW** | teks "Chicken Waffle" tegak hanya di CCW; skor 6.588 vs 3.825 |
| `valid/182169108` | rot90 CW | skor 6.867 vs 2.630 |

`train/182169107` juga transposed tetapi dibuang lebih dulu sebagai kebocoran.

**Penting:** rotasi tidak seragam. `valid/182169109` butuh arah berlawanan,
terkonfirmasi dua bukti independen. Rotasi seragam akan merusak gambar itu.
Gambar asli tidak diubah; hasilnya di `data/processed/images_fixed/`.

### 1.5 Kebocoran antar-split

Tiga pasang near-duplicate train↔test (dHash 0, 0, 1). **Salinan di train yang
dibuang**, test tetap utuh: `train/182169107`↔`test/114`, `train/182169171`↔
`test/227`, `train/182169179`↔`test/067`.

Alasan memilih sisi train: test set adalah objek evaluasi yang harus dijaga
keutuhannya. Setelah pembersihan, irisan nama file antar ketiga split = **0**.

### 1.6 Deduplikasi anotasi

Kriteria: kategori sama, bbox IoU > 0,85 **dan** mask IoU > 0,70. Syarat kedua
ditambahkan karena satu pasangan (`toy`, mask IoU 0,645) hanya bersinggungan —
tanpa filter mask, objek sah akan ikut terhapus.

| Split | Terdeteksi | Dibuang |
|---|---|---|
| train | 10 | 9 |
| valid | 4 | 4 |
| test | 18 | 18 |

Total 31 dibuang. Satu pasangan train sengaja dipertahankan (mask IoU 0,645).

### 1.7 Koreksi geometri

**Bbox diturunkan ulang dari poligon.** Satu anotasi di `test/124_jpg`
(`window-other`) punya bbox 180×574 sedangkan poligonnya hanya 61,9×60,9 —
deviasi 281 px. Setelah perbaikan, rasio luas poligon terhadap `area` = 1,0000.
Efek samping yang diinginkan: `area` kini berisi luas **mask**, bukan luas bbox,
sesuai yang dibutuhkan COCO untuk rentang small/medium/large.

**Clipping koordinat.** 1.891 anotasi punya titik yang menyentuh tepi (mis.
`x = 640.0` pada lebar 640), digeser ke `W − 0,01`. Besar pergeseran median
0,005 px, maksimum 0,005 px — kosmetik. Setelah itu poligon dan bbox OOB = 0.

### 1.8 Kebijakan resize

```yaml
target_long_side: 640
keep_aspect_ratio: true
padding: letterbox
upscale_limit: 2.0
min_object_px: 4
eval_resize: letterbox_640
```

**Dataset tidak di-resize fisik.** Resize di dataloader bersifat reproducible dan
tidak mengunci satu resolusi untuk semua model. Kebijakannya tetap eksplisit dan
seragam, karena test rentangnya 149 px–10,3 MP sedangkan train/valid sudah
640×480.

### 1.9 Distribusi kelas akhir

Top-10 tetap didominasi `wall-other-merged` (1.712), `table-merged` (705),
`person` (691), `chair` (632). Kelas `<5` instance: 3 · `<10`: 12 · `<50`: 30.
Empat kelas hilang di test: `wall`, `tree-merged`, `shelf`, dan satu kelas stuff
kecil. Long tail **tidak diperbaiki** — itu karakteristik data, bukan cacat;
penanganannya lewat pelaporan AP per kelompok frekuensi.

---

## 2. Split 5-fold

`k=5`, **seed 42**, `w_size=2.0`, hanya di atas train+valid (547 = 387+160) →
±438 latih / ±109 validasi per fold. Test 359 gambar tidak pernah disentuh.

Stratified per kelas, dengan penimbangan untuk kelas minoritas. Hasil:
**7 kelas tidak pernah muncul di train+valid** — kelas itu otomatis bernilai 0
pada AP di semua fold, sehingga menekan rata-rata kedua arm secara setara
(perbandingan antar-arm tetap sah). Daftar kelasnya tercatat di `folds.json`
pada kunci `classes_absent_in_trainval`.

---

## 3. Pilot Modal.com

Bukan training penuh: 2 run × 10 epoch, YOLO26n-seg, fold 0. Tujuannya mengukur
waktu, biaya, dan menguji jalur artefak.

| Item | Hasil |
|---|---|
| GPU | NVIDIA L40S, 47,7 GB, 214 TFLOPS fp16 |
| Waktu training | 61,2 s (pretrained) / 58,2 s (scratch) |
| Waktu per epoch | 6,12 s / 5,82 s |
| VRAM puncak | 5,44 GB / 5,32 GB |
| Parameter | 3.126.280 |
| **Biaya terukur** | **$0,0642** untuk kedua run |

Tarif L40S **$1,95/jam**. Ekstrapolasi terukur: YOLO26n-seg 100 epoch =
0,17 GPU-jam/run → 5 fold × 2 arm = **$3,32**.

### 3.1 Temuan yang mengubah protokol

Arm pretrained sudah mencapai mAP50-95 0,146 pada epoch 8. Arm scratch
**0,000 di kesepuluh epoch**, sementara loss tetap turun (seg 11,67 → 5,06) —
jadi pipeline tidak rusak, model memang belum konvergen.

**Keputusan:** scratch dinaikkan ke **200 epoch**, pretrained tetap 100. Ini
membuat anggaran epoch kedua arm tidak sama, sehingga rasio pretrained:scratch
yang dilaporkan adalah **batas bawah**, bukan batas atas — asimetri ini
menguntungkan scratch. Dicatat sebagai keterbatasan.

Catatan teknis: `modal volume get` gagal saat tujuan berupa folder di Windows
(`Permission denied`). Solusinya unduh per-file dengan nama eksplisit.

---

## 4. Training penuh YOLO26n-seg

10/10 run selesai, `status: complete`. Kedua arm identik kecuali inisialisasi
bobot.

| Parameter | Pretrained | Scratch |
|---|---|---|
| Bobot awal | `yolo26n-seg.pt` (COCO) | `.yaml` (acak) |
| Epoch | 100 | 200 |
| `lr0` | 1,35e-4 | 2,7e-4 |
| `warmup_epochs` | 3 | 5 |
| `close_mosaic` | 10 | 20 |
| `batch` / `imgsz` | 16 / 640 | 16 / 640 |
| `optimizer` | AdamW | AdamW |
| `workers` / `seed` | 8 / 0 | 8 / 0 |
| `deterministic` | True | True |

Head klasifikasi: **33 dari 70 baris di-remap dari bobot COCO** berdasarkan nama
kelas (`cls_remap=True`, default Ultralytics 8.4.213). Dicatat agar klaim "head
di-reinit penuh" tidak diulang. Arm scratch tidak mendapat remap apa pun.

### 4.1 Hasil per fold

Mask mAP pada validation fold masing-masing (107–114 gambar/fold), diambil dari
epoch terbaik tiap run menurut `artifacts/csv_*.csv`:

| Fold | Pretrained mAP50-95 | Pretrained mAP50 | Scratch mAP50-95 | Scratch mAP50 | Epoch terbaik |
|---|---|---|---|---|---|
| 0 | 0,2416 | 0,3566 | 0,0194 | 0,0414 | pre 84, scr 174 |
| 1 | 0,2027 | 0,3123 | 0,0269 | 0,0616 | pre 77, scr 168 |
| 2 | 0,1527 | 0,2577 | 0,0216 | 0,0525 | pre 67, scr 144 |
| 3 | 0,1889 | 0,2717 | 0,0269 | 0,0545 | pre 82, scr 154 |
| 4 | 0,2006 | 0,3079 | 0,0237 | 0,0498 | pre 80, scr 166 |
| **Mean** | **0,1973** | **0,3012** | **0,0237** | **0,0520** | — |
| SD | 0,0319 | 0,0387 | 0,0033 | 0,0073 | — |

**Rasio pretrained : scratch = 8,32× pada mAP50-95.** Selisih 0,1736 dengan SE
gabungan ±0,0143 — jauh di luar noise antar-fold. Konsisten dengan hipotesis
hari 1: dengan ±450 gambar latih per fold, model 3,1 juta parameter tidak mampu
mempelajari representasi dari nol, bahkan dengan 200 epoch.

Pretrained mencapai puncak di epoch 67–84 lalu menurun; scratch masih naik
perlahan sampai epoch 144–174. Angka di tabel adalah puncak, bukan epoch
terakhir — pada epoch terakhir pretrained turun ke rata-rata 0,1819.

### 4.2 Waktu dan biaya

| Arm | Rata-rata per run |
|---|---|
| Pretrained (100 ep) | 459,7 s (7,66 menit) |
| Scratch (200 ep) | 1139,6 s (19,0 menit) |

Total GPU-time 2,22 jam, wall clock orkestrasi **36,5 menit**, peak VRAM
5,94–6,17 GB dari 47,7 GB (12–13% utilisasi). Biaya **±$6,6** (L40S $5,96).
Estimasi pra-run $5,2, realisasi $6,6 (+27%) karena orchestrator terpanggil dua
kali dan startup container. Total akun hari itu **$9,0701**.

VRAM rendah mengonfirmasi pilot: L40S jauh lebih dari cukup untuk YOLO, tidak
perlu A100.

### 4.3 Makro vs mikro

Ultralytics melaporkan **rata-rata makro antar kelas**, bukan pooling instance.
Terbukti dari log fold 0: rata-rata kolom per-kelas sama persis dengan baris
`all`. Konsekuensinya kelas ber-1 instance berbobot sama dengan `person` ber-56
instance.

Setelah dihitung ulang dengan pooling: **mikro P = 0,4104, R = 0,4405,
F1 = 0,4250** (GT 5.948 instance, TP 2.620, prediksi 6.384, FP 3.764).

Makro P lebih tinggi dari mikro karena 11–19 kelas per fold mendapat P ≈ 1
dengan R = 0 — model tidak memprediksi apa pun untuk kelas itu, sehingga
presisinya "sempurna" tapi tidak berguna. Batas keabsahan angka mikro: tiap
kelas dievaluasi pada ambang confidence F1-nya sendiri, jadi pooling ini
menggabungkan beberapa operating point — indikator realistis, bukan evaluasi
satu ambang.

---

## 5. Peningkatan skor YOLO26n → YOLO26m

Semua angka: mask mAP50-95. Setup: subset 24 kelas (`fix22`, turunan dataset
bersih), fold 0, val 107 gambar, seed 0, RTX 4060 Laptop 8 GB — bukan Modal.
Subset itu tidak disertakan; dibuat ulang dari dataset bersih.

| Run | Model | img | epoch | mAP50-95 | mAP50 |
|---|---|---|---|---|---|
| baseline | yolo26n-seg (3,1M) | 512 | 12 | 0,1766 | — |
| **final** | **yolo26m-seg (27,1M)** | **640** | **24** | **0,3517** | **0,4958** |

**+0,1751 (+99%).** 22 kelas naik, 2 turun (`book` 0,057→0,016, `bottle`
0,380→0,279).

### 5.1 Dekomposisi

Semua pada setting lain identik, 6 epoch:

| Perubahan | mAP50-95 | Efek |
|---|---|---|
| n@512 (acuan) | 0,1495 | — |
| s@512 (11,5M) | 0,2356 | **+0,086** |
| m@512 (27,1M) | 0,2446 | +0,009 |
| l@512 (31,5M) | 0,2452 | +0,0006 |
| m@640 | 0,2599 | **+0,015** |
| m@640, 24 epoch | 0,3517 | **+0,092** |

Urutan pengaruh: **skala model ≫ jumlah epoch > resolusi**. `l` tidak dipakai:
seri dengan `m` (0,2452 vs 0,2446) tetapi 2× lebih lambat.

### 5.2 Yang dicoba dan tidak membantu

| Percobaan | Hasil |
|---|---|
| `mask_ratio=2` | 0,2510 vs 0,2599 — lebih buruk, 3× lebih lambat |
| TTA saat validasi | 0,2599 → 0,2599, tidak berefek untuk seg |
| Evaluasi di imgsz 768 | 0,2484 — turun karena mismatch dengan training 640 |
| `mosaic=0, erasing=0` | −0,007 |
| `imgsz=384` | −0,021 |
| `workers=2/3/4` | maksimal 3% (Windows spawn mahal) |

Percepatan yang berhasil (pengukuran mesin, bukan trik ML): power plan Balanced →
High Performance 34,0 → 19,4 s/epoch; `cache="disk"` 19,4 → 16,8 s/epoch;
`warmup_epochs` 3 → 0 membuat kurva naik 1 epoch lebih awal. Power plan
dikembalikan ke Balanced setelah setiap run.

### 5.3 Setting final

```
model    yolo26m-seg.pt
data     nc=24 (subset fix22)
epochs   24      imgsz 640     batch 8
optimizer AdamW, lr0=5e-4, lrf=0.01, warmup_epochs=0
cls=1.0, copy_paste=0.3, close_mosaic=3
workers=0, cache=disk, seed=0
```

Waktu 27,3 menit (68,4 s/epoch), inference 18,0 ms/gambar.

Kelas tersisa lemah: `book` 0,016 · `cabinet-merged` 0,043 · `light` 0,062 ·
`blanket` 0,111 · `spoon` 0,111 · `paper` 0,154.

### 5.4 Keterbatasan

- Satu fold, satu seed. Rentang antar seed ±0,04 pada model nano, jadi kenaikan
  +0,175 jauh di luar noise — tetapi tabel 5-fold resmi tetap perlu dijalankan
  ulang dengan konfigurasi ini.
- Arm scratch belum diuji dengan model `m`; hasil lama (0,0234) memakai nano.
- Protokol mengunci imgsz 640 — konfigurasi ini sudah patuh.

---

## 6. Keputusan hari ini

| # | Keputusan | Alasan |
|---|---|---|
| 1 | Buang `indoor-objects`, `wallet`, `trash-bin`; merge `scissor`, `st-` | `indoor-objects` bukan kelas; dua lainnya domain produk, bukan instance |
| 2 | Buang sisi **train** pada kebocoran, test tetap utuh | Test adalah objek evaluasi |
| 3 | Dedup anotasi butuh bbox **dan** mask IoU | Tanpa filter mask, objek sah ikut terhapus |
| 4 | Rotasi per-gambar, bukan seragam | `valid/182169109` butuh arah berlawanan |
| 5 | Tidak resize fisik, hanya kebijakan letterbox 640 | Reproducible, tidak mengunci satu resolusi |
| 6 | Scratch 200 epoch, pretrained 100 | Scratch 0,000 di 10 epoch; asimetri menguntungkan scratch |
| 7 | Laporkan mikro P/R/F1 sebagai pelengkap makro | Makro mengangkat P karena kelas ber-P≈1,R=0 |
| 8 | Ganti n→m, imgsz 640, 24 epoch | Skala model pengaruh terbesar; `l` seri tapi 2× lebih lambat |

---

## 7. Berkas yang disimpan

Hanya yang tidak bisa diringkas ke teks:

| Berkas | Isi |
|---|---|
| `data/processed/categories.json` | 70 kategori final + flag `isthing` — ruang label yang dipakai semua model |
| `splits/folds.json` | 5-fold stratified seed 42 — pembagian yang dipakai 10 run |
| `artifacts/csv_*.csv` | `results.csv` per run (10 berkas) — sumber semua angka mAP |
| `scripts/preprocess.py` | Menghasilkan dataset bersih dari dataset mentah |
| `scripts/make_folds.py` | Menghasilkan `folds.json` |

Sengaja tidak disertakan karena bisa dibangkitkan ulang: `instances_*.json`,
`manifest.csv`, `change_log.json`, `class_distribution_final.csv`,
`resize_policy.yaml`, `images_fixed/`, log panjang, dan `local_runs/`.

---

## 8. Rujukan

- Temuan EDA yang jadi dasar semua perbaikan: `notebooks/01_eda.ipynb`
- Keputusan model & hipotesis: `reports/LOGBOOK_HARI_1.md`
- Dataset: https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5
