# Logbook — Hari 1

**Tanggal:** 12 November 2026

**Lingkup hari ini:** audit dataset, pemilihan komposisi model, perumusan hipotesis

---

## 0. Catatan tentang sumber tanggal

Tidak semua isi logbook ini punya bukti tanggal yang sama kuatnya. Supaya
jujur, saya pisahkan:

| Bagian | Dasar penempatan |
|---|---|
| Eksplorasi data | **Terverifikasi.** Dataset diunduh 12 Nov 19:39; berkas log analisis EDA bertanggal 12 Nov 21:49 |
| Pemilihan model | **Rekonstruksi.** Tercatat di `AGENT.md` §4, tetapi berkas itu dokumen hidup yang terakhir diubah 24 Nov. Penempatan di hari 1 berdasarkan ingatan pengerjaan, bukan timestamp |
| Hipotesis awal | **Rekonstruksi.** Sama seperti di atas — `AGENT.md` §9, tanpa timestamp sendiri |

Jam dan urutan menit tidak bisa direkonstruksi dan tidak saya karang. Yang bisa
dipertanggungjawabkan adalah isi keputusannya, bukan detik keputusannya.

---

## 1. Eksplorasi data

### 1.1 Yang dikerjakan

Audit menyeluruh atas dataset yang baru diunduh, mencakup delapan pemeriksaan:

| Pemeriksaan | Isi |
|---|---|
| Integritas data | kecocokan file di disk dengan entri JSON, duplikat `image_id`/`annotation_id`, anotasi orphan |
| Properti gambar | resolusi, ukuran, rasio aspek, kecerahan, kontras, ketajaman, saturasi, EXIF |
| Asal-usul split | pola nama file sumber, normalisasi resolusi, divergensi distribusi kelas |
| Duplikat & kebocoran | hash berkas, near-duplicate dHash lintas split |
| Statistik anotasi | jumlah instance per gambar, luas relatif mask, bucket ukuran, bbox aspect |
| Higiene kelas | frekuensi per kelas, Gini, entropi, kelas yang hilang di split tertentu |
| Stuff vs things | pemisahan 18 kelas stuff dari 52 things, dampaknya ke metrik |
| Anomali visual | verifikasi per-anotasi lewat overlay mask untuk kelas yang geometrinya mencurigakan |

Hasilnya disajikan sebagai satu notebook, `notebooks/01_eda.ipynb`: kode dan
outputnya berdampingan, seluruh gambar dirender di memori, dan angka di laporan
ini berasal dari sana.

### 1.2 Profil dataset

| Split | Gambar | Anotasi | Ukuran |
|---|---|---|---|
| train | 397 | 4.479 | 15,21 MB |
| valid | 166 | 2.022 | 6,38 MB |
| test | 359 | 5.344 | 35,63 MB |
| **Total** | **922** | **11.845** | **62,48 MB** |

75 kategori, lisensi CC BY 4.0, format COCO Segmentation. Resolusi median
640×480. Densitas 11,8 instance/gambar.

Sebagai pembanding COCO 2017: dataset ini 0,34% ukuran gambar latih COCO
(397 vs 118.287) tetapi 1,6× lebih padat instance-nya. Konsekuensi langsung:
setiap model akan overfit parah, dan itu batasan data, bukan batasan GPU.

### 1.3 Enam temuan yang mengubah rencana

| # | Temuan | Dampak |
|---|---|---|
| 1 | Split `test` berasal dari populasi berbeda, bukan split acak dari pool yang sama | "Domain shift" pada test asli tercampur perbedaan sumber data |
| 2 | Dataset mencampur adegan indoor panoptic dengan foto produk close-up | Sebagian anotasi bukan instance dalam pengertian COCO |
| 3 | 4 gambar berdimensi terbalik (JSON 640×480, file 480×640) | Mask tidak align, harus diperbaiki sebelum training |
| 4 | 3 pasang near-duplicate lintas train↔test | Kebocoran ringan, menginflasi skor test |
| 5 | 39,9% anotasi adalah stuff, bukan things | AP things-only wajib jadi metrik utama |
| 6 | 32 kelas <50 instance, 14 kelas <10 | mAP kelas minoritas tidak stabil secara statistik |

### 1.4 Bukti di balik temuan yang paling menentukan

**Split test bukan dari populasi yang sama.** Tiga bukti independen: nama file
sumbernya beda (`182169xxx` untuk train/valid, `1–360` untuk test), resolusi
test tidak dinormalisasi (0% berukuran 640×480, rentang 149 px – 10,3 MP),
dan divergensi distribusi kelasnya KL(train‖test) = **0,801** vs
KL(train‖valid) = **0,135** — selisih 6×.

Ini temuan yang paling mengubah cara pelaporan. Skor pada test asli tidak boleh
disebut "domain shift" murni, melainkan **cross-source transfer**.

**Ketidakseimbangan kelas ekstrem.** Gini **0,634**, entropi 5,13 bit dari
maksimum 6,21. `wall-other-merged` 756 instance di train vs `microwave` dan
`backpack` 3, `tv` 2. Ini yang memaksa 5-fold **stratified per kelas** — split
acak akan menghilangkan kelas dari fold validasi.

**Hampir 40% anotasi bukan instance.** Stuff 1.785 (39,9%) train, 734 (36,3%)
valid, 1.872 (35,0%) test. Median luas relatif stuff 3,86% vs things 1,46%.
Melaporkan mAP gabungan akan menghasilkan angka yang tidak bisa
dipertanggungjawabkan untuk tugas instance segmentation.

**Campuran domain.** `wallet`, `pan`, `trash-bin`, `toy` terverifikasi visual
sebagai foto produk studio — bukan adegan ruangan. Ketiganya menyumbang 575
anotasi train (12,8%) tetapi hampir hilang di test (`trash-bin` 153 train vs 4
test, `box` 95 vs 1).

**EXIF kosong di seluruh 922 file.** Metadata di-strip Roboflow. Konsekuensi:
lampiran EXIF untuk tugas hanya bisa berasal dari 40 foto sendiri.

### 1.5 Tindakan yang diputuskan

Delapan perbaikan wajib sebelum training, semuanya lahir dari hari ini:

1. Perbaiki 4 gambar transposed
2. Keluarkan 3 pasang near-duplicate lintas split dari sisi train, test tetap utuh
3. Buang `indoor-objects` (0 anotasi, artefak level proyek Roboflow)
4. Tandai 18 kelas stuff terpisah; hitung AP things-only
5. Deduplikasi anotasi (bbox IoU > 0,85 + mask IoU > 0,70)
6. Merge `scissor`→`scissors`, `st-`→`stairs`
7. Tetapkan kebijakan resize eksplisit, jangan resize fisik
8. 5-fold stratified per kelas

### 1.6 Implikasi ke protokol

- "Test set asli" dilaporkan sebagai **cross-source transfer**, ditulis di metodologi
- Efek domain shift bersih hanya terukur dari 40 foto sendiri
- AP things-only jadi metrik utama, AP gabungan pelengkap
- AP dilaporkan per kelompok frekuensi: head ≥100, mid 20–99, tail <20
- 3 kelas tersulit berbasis data: `remote` (9 instance, median luas 0,17%),
  `clock` (3; 0,19%), `knife` (12; 0,22%), dengan `light` (41; 0,28%) sebagai
  kandidat keempat
- Kandidat analisis kegagalan sudah tersedia sejak hari pertama: objek tipis
  (828 bbox aspect ekstrem), objek sangat kecil (persentil 25 luas relatif =
  0,52%, jadi seperempat anotasi berukuran di bawah itu), kelas tail, dan
  37 gambar berukuran <200 px di salah satu sisinya

---

## 2. Pemilihan komposisi model

### 2.1 Empat model final

| Slot | Model | Params | Keluarga | Implementasi |
|---|---|---|---|---|
| 1 | EVA-02-L + Cascade Mask R-CNN | ≈357M | Two-stage / proposal-based | MMDetection + backbone EVA |
| 2 | Swin-L + Mask2Former | ≈216M | Query-based transformer | Detectron2 `mask2former_swin-l` |
| 3 | ECSeg-L (EdgeCrafter) | 34M | DETR-style kompak hasil distilasi | Repo resmi EdgeCrafter, config `ecseg_l.yml` |
| 4 | YOLO26n-seg | ≈3M | One-stage NMS-free | `ultralytics` |

**Keluarga efektif hanya 3**, bukan 4: ECSeg-L dan Mask2Former sama-sama
query-based. Ini dicatat sebagai aturan agar laporan tidak mengklaim lebih dari
yang bisa dibuktikan.

Rentang parameter sengaja lebar — 3M sampai 357M — supaya pertanyaan
"seberapa besar pengaruh kapasitas model pada dataset sekecil ini" bisa dijawab,
bukan sekadar "model mana yang menang".

### 2.2 Ablasi bonus

**Swin-L + HTC++** (≈250M, 51,1 mAP) dibandingkan dengan Swin-L + Mask2Former.
Backbone dan pretraining identik, hanya head berbeda — perbandingan terkontrol
antara two-stage dan query-based. Dilaporkan terpisah, bukan model kelima.

### 2.3 Kandidat yang ditolak

Bagian ini sama pentingnya dengan yang dipilih, karena menunjukkan batasannya
dinyatakan secara eksplisit.

| Kandidat | Alasan ditolak |
|---|---|
| SwinV2-G + HTC++ | ≈3B parameter, melebihi batas 500M; arm scratch mustahil konvergen |
| Nexus / OneSparse (ConvNeXt-S) | 53,4 mAP dengan ≈50M, tetapi tidak ada rilis kode → tidak dapat direproduksi |
| TAPe+ML v3 | 58,4 mAP dengan <100k parameter, klaim luar biasa tanpa verifikasi independen dan kode tidak tersedia |
| ViT-H + Cascade (53,1) / HQ-SAM-H (49,9) | Melebihi batas 500M |
| Mask R-CNN R50 / Cascade R50 | Duplikasi keluarga two-stage tanpa tambahan informasi |
| MambaVision-B + Cascade (45,7; ≈33M) | Alternatif kalau butuh keluarga SSM, tapi mAP terlalu rendah untuk diprioritaskan |
| DiNAT-L + Mask2Former / EoMT | Duplikasi head dan keluarga dengan Swin-L + Mask2Former |

Dua kandidat berperforma tertinggi ditolak bukan karena jelek, melainkan karena
tidak dapat direproduksi — tidak ada kode publik atau tidak terverifikasi.
Untuk tugas yang dinilai dari bukti, itu alasan yang sah.

### 2.4 Catatan fairness

Sumber pretraining berbeda per model dan **tidak boleh disamakan**:

- EVA-02-L: ImageNet-22K + COCO
- Swin-L: ImageNet-22K + COCO
- ECSeg-L: COCO, plus varian Objects365 (angka 48,8 berasal dari checkpoint O365)
- YOLO26n-seg: COCO

Konsekuensi yang harus ditulis di bagian keterbatasan: "arm pretrained" bukan
satu perlakuan setara. Yang terukur adalah efek pretraining *pada sumber data
masing-masing*, bukan efek pretraining secara umum.

### 2.5 Matriks run

4 model × 2 arm × 5 fold = 40 run mentah, direalistiskan:

| Model | Pretrained | Scratch |
|---|---|---|
| YOLO26n-seg | 5 fold | 5 fold |
| ECSeg-L | 5 fold | 5 fold |
| Swin-L + Mask2Former | 5 fold | 1 split (lower bound) |
| EVA-02-L + Cascade | 5 fold | 1 split (lower bound) |

Ditambah ablasi Swin-L + HTC++ 5 fold pretrained. Total 37 run, ±44 GPU-hour.

Model besar dibatasi di arm scratch karena hasilnya sudah bisa diprediksi
(underfit) — dan itu justru salah satu hal yang ingin dibuktikan.

5-fold hanya di atas `train` + `valid` (547 gambar setelah cleaning), ±438 latih
/ ±109 validasi per fold. `test` 359 gambar tidak pernah disentuh.

---

## 3. Hipotesis awal

Ditulis **sebelum** training dijalankan. Tujuannya supaya nanti bisa dinilai
mana yang benar, mana yang meleset, dan mana yang salah karena alasan yang
salah.

### 3.1 Hipotesis utama

**H1 — Efek pretrained paling besar pada model kecil.**
YOLO26n-seg dan ECSeg-L akan menunjukkan selisih pretrained vs scratch paling
besar, karena kapasitasnya tidak cukup untuk mempelajari representasi dari
±450 gambar.

**H2 — Pada model besar, gap lebih didominasi kegagalan konvergensi.**
Untuk EVA-02-L dan Swin-L, selisih pretrained vs scratch bukan bukti bahwa
arsitektur butuh pretraining, melainkan bahwa arm scratch tidak sempat
konvergen. Harus dibahas sebagai konfounder, bukan sebagai temuan.

**H3 — ECSeg-L adalah uji paling tajam.**
EdgeCrafter secara eksplisit mengklaim ViT kompak butuh distilasi task-specific.
Arm scratch menguji klaim itu langsung: kalau klaimnya benar, ECSeg-L scratch
akan gagal jauh lebih parah daripada model besar scratch.

**H4 — Kegagalan terbesar pada objek tipis dan kecil.**
Cascade/Mask R-CNN dengan mask head 28×28 akan paling menderita pada objek
tipis; YOLO-seg bergantung pada resolusi prototype mask. Prediksi konkretnya:
kelas `remote`, `clock`, `knife` jadi yang terburuk, dan 828 bbox aspect ekstrem
serta 25% anotasi berluas <0,52% adalah sumber kegagalannya.

**H5 — Domain shift terbesar dari foto kondisi sulit.**
Selisih performa terbesar antara test set asli dan 40 foto sendiri akan muncul
pada foto cahaya redup dan backlight, bukan pada foto kondisi normal.

### 3.2 Yang sudah diketahui akan mengganggu

Bukan hipotesis, tapi harus disebut di keterbatasan:

- 41 gambar hanya berisi stuff dan 1 gambar tanpa anotasi → arm scratch tertekan,
  dan itu bukan kesalahan model
- `test` bukan populasi yang sama → skor test asli tidak bisa disebut domain shift murni
- Sumber pretraining berbeda per model → perbandingan antar-model tidak sepenuhnya setara
- Head tidak sepenuhnya acak: Ultralytics 8.4.213 memakai `cls_remap=True`
  default, sehingga 33/70 baris head mewarisi bobot COCO berdasarkan kecocokan
  nama kelas. Arm scratch tidak mendapat remap apa pun, jadi perbandingan antar-arm
  tetap valid — tapi ini harus didokumentasikan

---

## 4. Keputusan hari ini

| # | Keputusan | Alasan |
|---|---|---|
| 1 | Pakai dataset apa adanya, tapi bersihkan dulu lewat 8 tindakan | Integritasnya baik (0 file hilang, 0 duplikat id), masalahnya di kualitas anotasi |
| 2 | Laporkan test asli sebagai cross-source transfer | KL 0,801 vs 0,135 membuktikan beda populasi |
| 3 | AP things-only sebagai metrik utama | 39,9% anotasi bukan instance |
| 4 | 5-fold stratified per kelas, hanya di train+valid | 32 kelas <50 instance; test harus utuh |
| 5 | 4 model, 3 keluarga efektif, rentang 3M–357M | Menjawab pengaruh kapasitas, bukan hanya peringkat |
| 6 | Tolak kandidat tanpa kode publik | Tidak dapat direproduksi = tidak dapat dipertanggungjawabkan |
| 7 | Batasi arm scratch model besar ke 1 split | Hasilnya sudah bisa diprediksi; hemat 30+ GPU-hour |
| 8 | Tulis H1–H5 sebelum training | Supaya bisa dievaluasi nanti, bukan disesuaikan setelah tahu hasil |

---

## 5. Rujukan

- Analisis lengkap: `notebooks/01_eda.ipynb` (dapat dijalankan ulang, output tertanam)
- Dataset: https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5
