# Logbook — Hari 5

**Tanggal:** 23 November 2026

## Kendala dan keputusan

### 1. Mask2Former scratch tidak menghasilkan prediksi

Run awal `swinl35_scratch_f0_e150_b2_640_m20` mendapat AP mendekati nol dan `n_pred=0`. Pemeriksaan menemukan query embedding tidak terinisialisasi sesuai bobot COCO: 200 query hampir identik, dengan simpangan baku antarkueri `9,4e-6`. Grid threshold awal juga dimulai di 0,20, padahal skor model sekitar 0,06. Selain itu, `min_delta=5e-4` lebih besar daripada kenaikan AP sekitar `1e-5`, sehingga checkpoint terbaik tertahan di epoch 1.

**Keputusan:** inisialisasi query embedding dengan distribusi normal standar; perluas grid threshold scratch menjadi 0,01–0,40; simpan checkpoint setiap AP meningkat dan gunakan `min_delta` hanya untuk patience. LR backbone scratch dinaikkan dari 0,1× menjadi 1,0× dan gradient accumulation diset 2.

Setelah perbaikan, prediksi muncul (`max_fg` naik dari 0,06 ke 0,47) dan gradien stabil di kisaran 70–310. AP scratch tetap rendah (0,000747 pada epoch 68), sehingga run dihentikan manual. Pretrained mencapai AP validasi 0,3265 pada epoch 75 dan AP test 0,2118.

**Koreksi keputusan:** dugaan bahwa `decoder.level_embed` bernilai nol merupakan bug dibatalkan setelah pemeriksaan kode menunjukkan itu adalah inisialisasi resmi Mask2Former (`std=0`).

### 2. YOLO26m scratch 35 kelas belum memuaskan pada percobaan awal

Percobaan `k35_m_sf0` tidak memuaskan; `k35_m_sf0_c2` dilanjutkan dengan tuning. Run `k35_m_sf0_c3` menjadi run scratch terpilih dengan mask mAP50–95 0,1016. Arm pretrained `k35_m_pf0` memperoleh 0,3347.

**Keputusan perubahan jumlah kelas:** pada 23 November, protokol 24 kelas tidak lagi digunakan; eksperimen beralih ke **35 kelas (24 things + 11 stuff)**. Alasannya, evaluasi 35 kelas menunjukkan mean AP things 0,3440—0,0077 lebih rendah daripada acuan 24 kelas 0,3517—sekaligus menghasilkan mean AP 0,3143 pada 11 kelas stuff. Pemeriksaan label juga memastikan 35/35 kelas muncul dan tidak ada kelas kosong. Dengan demikian, tambahan 11 kelas tidak menimbulkan penurunan berarti pada things dan memberi cakupan kelas yang lebih lengkap. Keputusan ini berlaku untuk eksperimen dan pelaporan berikutnya; AP things dan stuff dilaporkan terpisah.

Untuk arm scratch, gunakan `k35_m_sf0_c3` sebagai checkpoint final karena skornya tertinggi dalam rangkaian percobaan scratch 35 kelas.

Catatan protokol: rangkaian scratch merupakan warm restart berantai, bukan resume optimizer yang setara. Perbandingan pretrained–scratch perlu disertai keterbatasan ini.

### 3. EVA-02-L tidak melatih RPN

`mmdet 3.3.0` mengembalikan `loss_rpn_cls` dan `loss_rpn_bbox` sebagai list tensor. Kode lama hanya menjumlahkan nilai yang lolos `torch.is_tensor(v)`, sehingga kedua loss RPN tidak masuk backward. Bukti: log training tidak memuat loss RPN, gradien `rpn_head` kosong pada 6/6 parameter, sementara `roi_head` memiliki gradien pada 60/60 parameter. RPN recall tercatat 0,36. Loss total tetap turun karena tiga stage Cascade belajar, jadi loss total tidak cukup untuk mendeteksi masalah ini.

Pada uji overfit delapan gambar dan 300 iterasi, perbaikan menaikkan bbox AP dari 0,1668 ke 0,8544 dan mask AP dari 0,1891 ke 0,6740.

**Keputusan:** batalkan seluruh angka AP EVA dari run yang memakai loss RPN lama dan ulang training dari awal dengan penjumlahan loss list yang benar. Ukuran waktu, VRAM, dan biaya yang tidak bergantung pada checkpoint tersebut tetap dapat digunakan.
