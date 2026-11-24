"""Training pilot YOLO26n-seg di Modal: build dataset YOLO dari COCO + latih + catat log."""
import json
import os
import shutil
import sys
import tarfile
import time

import modal

app = modal.App("kompis-yolo")

vol = modal.Volume.from_name("kompis", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "procps")
    .pip_install(
        "ultralytics>=8.3.0",
        "pycocotools",
        "opencv-python-headless",
        "pandas",
        "pillow",
    )
    .env({"YOLO_VERBOSE": "True", "PYTHONUNBUFFERED": "1"})
)

VOL = "/vol"
DS = f"{VOL}/ds"


class Tee:
    """Tulis ke stdout dan ke file log sekaligus."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def ensure_data():
    """Ekstrak tar dataset sekali saja."""
    marker = f"{DS}/.extracted"
    if os.path.exists(marker):
        return "sudah ada"
    os.makedirs(DS, exist_ok=True)
    with tarfile.open(f"{VOL}/kompis_data.tar.gz") as tf:
        tf.extractall(f"{VOL}/_tmp_extract")
    # tar berisi train/, valid/, test/, data/processed/, splits/
    src = f"{VOL}/_tmp_extract"
    for name in os.listdir(src):
        shutil.move(os.path.join(src, name), os.path.join(DS, name))
    os.rmdir(src)
    open(marker, "w").close()
    vol.commit()
    return "diekstrak"


def build_yolo(fold: int) -> str:
    """Bangun struktur YOLO (images/ + labels/) untuk satu fold dari COCO bersih."""
    out = f"{DS}/yolo/f{fold}"
    if os.path.exists(f"{out}/data.yaml"):
        return out

    cats = json.load(open(f"{DS}/data/processed/categories.json", encoding="utf-8"))
    names = [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
    folds = json.load(open(f"{DS}/splits/folds.json", encoding="utf-8"))["folds"][fold]

    all_anns, all_imgs = [], {}
    for split in ["train", "valid"]:
        d = json.load(open(f"{DS}/data/processed/instances_{split}.json", encoding="utf-8"))
        all_anns += d["annotations"]
        for im in d["images"]:
            all_imgs[im["id"]] = im

    for part in ["train", "val"]:
        _write_part(out, part, folds[f"{part}_ids"], all_imgs, all_anns, names)

    with open(f"{out}/data.yaml", "w", encoding="utf-8") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")
    vol.commit()
    return out


def _write_part(root, part, ids, by_id, anns, names):
    img_dir = os.path.join(root, "images", part)
    lbl_dir = os.path.join(root, "labels", part)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    name_to_idx = {n: i for i, n in enumerate(names)}
    cat_id_to_name = {i + 1: n for i, n in enumerate(names)}
    per_img = {}
    for a in anns:
        per_img.setdefault(a["image_id"], []).append(a)
    for i in ids:
        im = by_id[i]
        src = os.path.join(DS, im["file_name"])
        dst = os.path.join(img_dir, os.path.basename(im["file_name"]))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        w, h = im["width"], im["height"]
        lines = []
        for a in per_img.get(i, []):
            cls_name = cat_id_to_name.get(a["category_id"])
            if cls_name is None:
                continue
            cls = name_to_idx[cls_name]
            for poly in a["segmentation"]:
                pts = [round(float(v), 6) for v in poly]
                if len(pts) < 6:
                    continue
                norm = []
                for k in range(0, len(pts), 2):
                    norm.append(min(max(pts[k] / w, 0.0), 1.0))
                    norm.append(min(max(pts[k + 1] / h, 0.0), 1.0))
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in norm))
        with open(os.path.join(lbl_dir, os.path.splitext(os.path.basename(im["file_name"]))[0] + ".txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


@app.function(image=image, gpu="L40S", volumes={VOL: vol}, timeout=3600)
def train_pilot(fold: int = 0, epochs: int = 10, batch: int = 16, imgsz: int = 640,
                arm: str = "pretrained", model_name: str = "yolo26n-seg"):
    tag = f"yolo_{model_name}_{arm}_f{fold}"
    log_path = f"{VOL}/logs/{tag}.log"
    os.makedirs(f"{VOL}/logs", exist_ok=True)
    tee = Tee(log_path)
    sys.stdout = tee

    t_start = time.time()
    print(f"=== {tag} | arm={arm} epochs={epochs} batch={batch} imgsz={imgsz} ===", flush=True)
    print(f"[data] {ensure_data()}", flush=True)
    data_yaml = build_yolo(fold)
    print(f"[data] yaml = {data_yaml}", flush=True)

    import torch
    from ultralytics import YOLO

    torch.manual_seed(0)
    if arm == "pretrained":
        model = YOLO(f"{model_name}.pt")
        print(f"[model] pretrained {model_name}.pt", flush=True)
    else:
        model = YOLO(f"{model_name}.yaml")
        print(f"[model] scratch {model_name}.yaml (random init)", flush=True)

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[model] parameters: {n_params:,}", flush=True)

    run_dir = f"{VOL}/runs/{tag}"
    t_train = time.time()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0,
        workers=8,
        seed=0,
        deterministic=True,
        project=f"{VOL}/runs",
        name=tag,
        exist_ok=True,
        plots=True,
        val=True,
        verbose=True,
    )
    train_seconds = time.time() - t_train

    best = f"{run_dir}/weights/best.pt"
    metrics = {}
    try:
        m = YOLO(best).val(data=data_yaml, device=0, verbose=False)
        metrics = {
            "mAP50_95_mask": float(m.seg.map),
            "mAP50_mask": float(m.seg.map50),
            "mAP50_95_box": float(m.box.map),
            "mAP50_box": float(m.box.map50),
        }
    except Exception as e:
        metrics = {"val_error": f"{type(e).__name__}: {e}"}

    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    total = time.time() - t_start
    summary = {
        "tag": tag, "arm": arm, "fold": fold, "epochs": epochs,
        "batch": batch, "imgsz": imgsz, "params": n_params,
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(total, 1),
        "sec_per_epoch": round(train_seconds / epochs, 2),
        "peak_vram_gb": peak_gb,
        "gpu": torch.cuda.get_device_name(0),
        **metrics,
    }
    with open(f"{VOL}/runs/{tag}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {json.dumps(summary)}", flush=True)
    tee.flush()
    sys.stdout = sys.__stdout__
    vol.commit()
    return summary
