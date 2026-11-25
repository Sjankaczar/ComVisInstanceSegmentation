# Logbook — Hari 4

**Tanggal:** 22 November 2026
**Lingkup hari ini:** mengukur kelayakan ECSeg-L di GPU lokal, memvalidasi alur training Mask2Former, dan mengunci batas VRAM serta metode gate.

---

## 0. Catatan sumber dan artefak yang disimpan

Angka hari ini diringkas dari `reports/GATE_REPORT.md`, laporan pengukuran dan gate yang menjadi dasar keputusan untuk pekerjaan berikutnya. Berkas yang disimpan adalah laporan keputusan final tersebut. Skrip pilot/gate sementara, hasil uji LR yang gagal, serta checkpoint prototipe Mask2Former tidak dimasukkan: semuanya merupakan eksperimen antara dan tidak menjadi artefak final yang dipakai pada hasil akhir.

---

## 1. Pengukuran ECSeg-L di lokal

Perkiraan awal bahwa ECSeg-L memerlukan 28–45 jam ternyata berasal dari ekstrapolasi ECSeg-S, bukan pengukuran langsung. Pengujian di RTX 4060 Laptop 8 GB menunjukkan ECSeg-L batch 2 berjalan sekitar **0,729 detik/iterasi**, hampir sama dengan ECSeg-S (**0,706 detik/iterasi**). Satu epoch ECSeg-L memerlukan sekitar 2 menit 40 detik.

Batch 4 menyentuh sekitar 7,9 GB VRAM dan menyebabkan spill ke RAM sistem. Waktu iterasi memburuk 10–16 kali. Karena itu, batas aman training ditetapkan **batch 2**; ECSeg-L layak dilatih lokal.

## 2. Gate G5 dan keputusan protokol

Gate overfit ECSeg pada delapan gambar (361 anotasi), tanpa augmentasi, mencapai AP@0.5 **0,046** setelah 40 epoch atau 160 iterasi. Pemeriksaan GT sebagai prediksi menghasilkan AP 1,000, dan spot-check 5/5 anotasi cocok. Jadi hasil rendah itu bukan bukti kerusakan label. Diagnosisnya: model DETR-style memerlukan jauh lebih banyak iterasi daripada yang dicakup gate tersebut.

**Keputusan:** gate berikutnya harus berbasis jumlah iterasi (sekurangnya 2.000 iterasi pada empat gambar), bukan jumlah epoch.

## 3. EVA-02-L tidak dijalankan lokal

Lingkungan Windows lokal tidak mendukung dependensi EVA yang diperlukan: `mmcv==1.7.1` tidak menyediakan wheel untuk kombinasi Torch 2.7/cu126, sementara Detectron2 perlu dibangun dari source dengan MSVC. EVA dipindahkan ke Modal/Linux.

## 4. Keputusan yang dibawa ke eksperimen berikutnya

- Gunakan **ECSeg-L**, bukan ECSeg-S: skor validasi sebelumnya lebih tinggi, jadwalnya lebih pendek, dan pengukuran menunjukkan kecepatannya setara.
- Pertahankan batch **2** untuk mencegah spill VRAM.
- Evaluasi gate ECSeg berdasarkan iterasi yang benar-benar dijalankan.
- Jalankan EVA pada lingkungan Linux Modal.

Laporan pengukuran lengkap beserta tabel dan rincian gate ada di [`GATE_REPORT.md`](GATE_REPORT.md).
