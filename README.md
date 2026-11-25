# ComVisInstanceSegmentation

Segmentasi instance pada lingkungan indoor — tugas besar Computer Vision.

Repo ini berisi pekerjaan hari pertama sampai ketiga: audit dataset, pemilihan
komposisi model, hipotesis, preprocessing, training awal, dan fine-tune ECSeg-L.

```
notebooks/01_eda.ipynb         analisis EDA bereksekusi, output tertanam
reports/LOGBOOK_HARI_1.md      hari 1: EDA, pemilihan model, hipotesis
reports/LOGBOOK_HARI_2.md      hari 2: preprocessing, fold, pilot, training, skor
reports/LOGBOOK_HARI_3.md      hari 3: ECSeg-L 36 epoch + 8 eksperimen
reports/LOGBOOK_HARI_4.md      hari 4: gate ECSeg-L, batas VRAM, dan keputusan Modal
reports/LOGBOOK_HARI_5.md      hari 5: debug Mask2Former, YOLO 35 kelas, diagnosis RPN EVA
local_runs/k35_m_pf0/         run YOLO26m pretrained (checkpoint, log, CSV, args)
local_runs/k35_m_sf0_c3/      run YOLO26m scratch final (checkpoint, log, CSV, args)
scripts/train_k35.py          skrip training YOLO 35 kelas
reports/ecseg_l/               kurva, AP per kelas, ablasi, matriks konfusi, config
data/processed/categories.json 70 kategori final + flag stuff/things
splits/folds.json              5-fold stratified (seed 42)
artifacts/csv_*.csv            results.csv per run, sumber semua angka mAP
scripts/preprocess.py          dataset mentah -> dataset bersih
scripts/make_folds.py          train+valid -> folds.json
modal/train_yolo.py            skrip training di Modal.com
docs/                          keterangan dataset
```

Gambar dataset dan berkas turunan (`instances_*.json`, `manifest.csv`,
`images_fixed/`) tidak disertakan. Semuanya dibangkitkan ulang oleh
`python scripts/preprocess.py` setelah dataset diunduh.

## Dataset

Dataset tersedia publik di Roboflow Universe:

**https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5**

922 gambar, anotasi COCO Segmentation, lisensi CC BY 4.0.

Berkas gambarnya **tidak di-commit ke repo ini** — 62 MB gambar tidak layak
masuk riwayat git. Yang di-commit hanya analisisnya. Unduh dataset dari tautan
di atas (format COCO Segmentation), lalu letakkan folder `train/`, `valid/`,
`test/` di mana saja dan arahkan notebook ke situ.

## Menjalankan notebook

```bash
pip install numpy pandas matplotlib opencv-python pillow jupyter
set DATASET_DIR=C:\path\ke\dataset
jupyter lab notebooks/01_eda.ipynb
```

`DATASET_DIR` harus berisi `train/`, `valid/`, `test/` beserta JSON COCO-nya.
Kalau variabel itu tidak diset, notebook mencari folder `train/` ke atas dari
direktori kerja.
