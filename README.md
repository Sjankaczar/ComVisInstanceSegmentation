# ComVisInstanceSegmentation

Segmentasi instance pada lingkungan indoor — tugas besar Computer Vision.

Repo ini berisi pekerjaan hari pertama: audit dataset, pemilihan komposisi
model, dan hipotesis yang akan diuji. Tahap berikutnya ditambahkan menyusul.

```
notebooks/01_eda.ipynb    analisis bereksekusi, output tertanam
reports/LOGBOOK.md        logbook: EDA, pemilihan model, hipotesis
docs/                     keterangan dataset
```

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

