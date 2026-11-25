# ComVisInstanceSegmentation

Proyek segmentasi instance untuk lingkungan indoor menggunakan Panoptic Indoor Segmentation v5 dan foto ruangan yang dianotasi sendiri.

## Isi proyek

- `REPORT_TUGAS.md` — laporan hasil, protokol, keterbatasan, analisis kegagalan, dan referensi artefak.
- `logbook.md` — ringkasan kegiatan, kendala, keputusan, dan hipotesis awal.
- `Training Model.ipynb` — kode dan catatan training model pretrained serta scratch.
- `Evaluasi Model.ipynb` — evaluasi test set asli dan sendiri, analisis domain shift, studi kasus kegagalan, dan demo inferensi.
- `eda.ipynb` — pemeriksaan distribusi dan integritas dataset sumber.
- `reports/own_test_eval/` — 40 foto test set sendiri, metadata EXIF, hasil evaluasi, grafik, dan panel kegagalan.
- `reports/training_evidence/` — log, konfigurasi, metrik per epoch, dan bukti training ECSeg-L, Swin-L, serta EVA.
- `reports/figures/` — gambar dan grafik yang dirujuk oleh laporan.
- `scripts/`, `modal/`, `splits/`, dan `data/processed/` — skrip preprocessing/training, konfigurasi Modal, definisi fold, serta kategori dataset.

## Dataset

Dataset sumber: [Panoptic Indoor Segmentation v5 di Roboflow Universe](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5). Dataset berisi 922 gambar dengan anotasi COCO Segmentation dan lisensi CC BY 4.0. Gambar dataset sumber tidak disimpan di proyek; unduh ekspor COCO dari sumber tersebut untuk menjalankan ulang EDA atau training.

Foto own-test dan metadata yang menyertainya berada di `reports/own_test_eval/`. Evaluasi own-test yang dilaporkan hanya mencakup kelas yang memiliki anotasi bersama, bukan seluruh 35 kelas protokol training. Rincian batasan dan protokol ada di `REPORT_TUGAS.md`.

