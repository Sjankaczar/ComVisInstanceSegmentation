# Logbook — Kendala dan Keputusan

**Tugas:** Segmentasi instance pada lingkungan indoor nyata  
**Periode:** 12–24 November 2026

Logbook ini merangkum kegiatan, kendala, dan keputusan utama. Hipotesis awal ditempatkan di depan agar dapat dibandingkan dengan hasil eksperimen.

## 15 November — Hipotesis sebelum training

Sebelum training utama, hipotesis yang akan diuji adalah:

1. **Pretrained lebih membantu pada data terbatas.** Selisih pretrained dan scratch diperkirakan besar, terutama pada model kecil.
2. **Scratch pada model besar berisiko belum konvergen.** Skor rendah tidak otomatis berarti arsitekturnya buruk; jumlah data dan lama training menjadi faktor pengganggu.
3. **ECSeg-L menjadi uji penting untuk manfaat pretraining.** EdgeCrafter mengklaim ViT kompak butuh distilasi task-specific. Model scratch diperkirakan tertinggal jauh dari model yang memakai bobot pretrained.
4. **Objek kecil dan tipis lebih sulit disegmentasi.** Kegagalan diperkirakan terkait ukuran objek dan kemampuan model menemukannya.
5. **Foto dengan pencahayaan sulit lebih terdampak pergeseran domain.** Foto redup atau backlight diperkirakan mengalami penurunan lebih besar.

Hipotesis tersebut merupakan prediksi, bukan kesimpulan. Hasil eksperimen dan keterbatasan tiap pengujian dirangkum pada tanggal terkait di bawah.

## 12 November — Audit dataset dan rancangan eksperimen

**Kegiatan:** Memeriksa integritas berkas, anotasi, distribusi kelas, resolusi, duplikasi, serta perbedaan train, valid, dan test. Memilih model dari beberapa keluarga segmentasi.

**Kendala:** Test bawaan berbeda sumber dan distribusinya dari train/valid; sebagian gambar memiliki dimensi yang tidak cocok dengan metadata; terdapat near-duplicate, kelas stuff, dan kelas berekor panjang.

**Keputusan:** Bersihkan anotasi dan gambar bermasalah sebelum training; jaga test tetap utuh; buat split validasi sendiri dari train+valid; laporkan things dan stuff secara terpisah. Skor test bawaan ditafsirkan sebagai transfer lintas sumber, bukan pergeseran domain murni.

## 15 November — Preprocessing, YOLO, dan alur training

**Kegiatan:** Menyelesaikan preprocessing, membuat lima fold, menguji pilot YOLO di Modal, lalu menjalankan training YOLO dan eksperimen tuning lokal.

**Kendala:** Pilot menemukan konfigurasi optimizer otomatis mengabaikan learning rate yang diminta dan bobot kelas pretrained sebagian dipetakan ulang. Beberapa job Modal gagal karena lifecycle container, timeout, API metrik, dan perbedaan versi Python. Run scratch singkat belum menghasilkan skor berarti.

**Keputusan:** Tetapkan optimizer dan learning rate secara eksplisit; catat pemetaan bobot kelas; lakukan verifikasi CPU dan rehearsal sebelum training panjang; jalankan job Modal secara terlepas dari proses lokal. Untuk hasil YOLO, gunakan model dan resolusi yang memberi peningkatan terukur, bukan mengandalkan threshold atau TTA yang tidak memperbaiki skor.

## 16 November — ECSeg-L 24 kelas dan eksperimen awal

**Kegiatan:** Fine-tune ECSeg-L pada label 24 kelas, menguji aspek gambar, threshold/NMS, resolusi, dan augmentasi, serta memeriksa AP per kelas dan matriks kebingungan.

**Kendala:** Perubahan threshold dan post-processing tidak memberi peningkatan yang meyakinkan. Kelas tertentu memiliki recall rendah; dugaan awal bahwa resolusi mask adalah penyebab utama tidak didukung oleh uji mask oracle.

**Keputusan:** Pertahankan augmentasi yang membantu, jangan memakai threshold per kelas yang dioptimasi pada validasi yang sama, dan prioritaskan diagnosis deteksi/recall. Uji rasio aspek native lebih lanjut karena hasil awalnya positif tetapi masih dalam batas noise.

## 22 November — Uji kelayakan lokal dan gate

**Kegiatan:** Mengukur kecepatan serta penggunaan VRAM ECSeg-L, menguji gate overfit, dan memeriksa kebutuhan lingkungan EVA-02-L.

**Kendala:** Estimasi durasi ECSeg-L dari model lebih kecil meleset jauh. Batch 4 menyebabkan spill VRAM ke RAM dan memperlambat iterasi. Gate 160 iterasi menghasilkan AP rendah meski label diperiksa dan terbukti konsisten. Dependensi EVA tidak cocok dengan Windows lokal.

**Keputusan:** Latih ECSeg-L lokal dengan batch 2; nilai gate model DETR-style berdasarkan jumlah iterasi, bukan epoch semata; jalankan EVA pada lingkungan Linux Modal.

## 23 November — Mask2Former, YOLO 35 kelas, dan koreksi EVA

**Kegiatan:** Memperbaiki run Mask2Former Swin-L, melanjutkan YOLO26m, memperluas label eksperimen, dan mengaudit alur training EVA.

**Kendala:** Mask2Former scratch awal tidak mengeluarkan prediksi akibat inisialisasi query, ambang evaluasi, dan aturan pemilihan checkpoint yang tidak cocok. Run scratch YOLO awal juga belum memuaskan. Pada EVA, loss RPN berbentuk list tidak ikut dijumlahkan untuk backward, sehingga RPN tidak belajar meski total loss turun.

**Keputusan:** Perbaiki inisialisasi dan evaluasi Mask2Former; pilih run scratch YOLO terbaik dari rangkaian percobaan. Pada 23 November, protokol beralih dari 24 menjadi **35 kelas: 24 things dan 11 stuff**, lalu laporkan kelompoknya terpisah. Batalkan skor EVA dari run yang RPN-nya tidak terlatih dan ulangi dengan akumulasi loss yang benar.

## 24 November — Training ECSeg-L 35 kelas dan audit hasil

**Kegiatan:** Menjalankan ECSeg-L pretrained dan scratch pada 35 kelas, memeriksa kurva per epoch, mengaudit konfigurasi dan checkpoint, serta melakukan anotasi test buatan sendiri.

**Kendala:** Mosaic dan mixup mendorong penggunaan memori hingga terjadi spill VRAM; epoch menjadi jauh lebih lambat. Saat mematikan mosaic, konfigurasi `mosaic_epoch` ternyata juga mengatur fase flat learning rate. Target AP scratch yang direncanakan sebelumnya tidak sesuai dengan bukti eksperimen model pembanding.

**Keputusan:** Matikan mosaic melalui probabilitasnya, bukan dengan mengubah jadwal flat learning rate; matikan mixup untuk run scratch yang stabil; simpan percobaan sebelumnya sebelum memulai ulang. Verifikasi checkpoint dan konfigurasi, lalu koreksi ekspektasi AP scratch agar sesuai skala hasil yang terukur.

## Ringkasan keputusan utama

- Gunakan evaluasi terpisah untuk things dan stuff, serta jelaskan bahwa test bawaan merupakan transfer lintas sumber.
- Bandingkan hasil dengan memperhatikan perbedaan split, label, dan anggaran training.
- Jalankan ECSeg-L lokal pada batch 2 dan EVA pada Linux Modal.
- Gunakan 35 kelas mulai keputusan 23 November dan laporkan things/stuff secara terpisah.
- Batalkan hasil EVA yang dihasilkan ketika RPN belum menerima gradien; gunakan hasil dari training ulang yang benar.
- Tafsirkan hasil scratch dengan mempertimbangkan jumlah data dan konvergensi, bukan hanya tren loss.
