"""Preprocessing dataset Panoptic Indoor Segmentation v5.

Menerapkan tindakan perbaikan §8 pada reports/eda/EDA_REPORT.md.
TIDAK melakukan training.

Output:
  data/processed/instances_{split}.json   COCO bersih per split
  data/processed/categories.json          kategori final + flag isthing
  data/processed/images_fixed/            gambar yang dirotasi (transposed fix)
  data/processed/manifest.csv             satu baris per gambar
  data/processed/change_log.json          audit semua yang dibuang/diubah
  data/processed/resize_policy.yaml       kebijakan resize
"""
import os, sys, json, glob, hashlib
from collections import Counter, defaultdict

import numpy as np
import cv2
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS = ["train", "valid", "test"]
OUT_DIR = os.path.join(ROOT, "data", "processed")
FIX_DIR = os.path.join(OUT_DIR, "images_fixed")
os.makedirs(FIX_DIR, exist_ok=True)

# ---------------------------------------------------------------- kebijakan
# §8#3: kategori dibuang / digabung
DROP_CATEGORIES = {"indoor-objects", "wallet", "trash-bin"}
# 'st-' terverifikasi visual = tangga (nama terpotong) -> digabung ke 'stairs'
MERGE_CATEGORIES = {"scissor": "scissors", "st-": "stairs"}

# §8#6 (diputuskan user): wallet + trash-bin dibuang, pan & toy dipertahankan.

# §8#2: near-duplicate lintas split -> buang salinan di TRAIN, test tetap utuh
LEAKAGE_DROP = {
    "train": ["182169107_jpg.rf.c9b1bbecf1474a5b36a6df197c30dfed.jpg",
              "182169171_jpg.rf.86d6a58acc7813643e19951675bb9dfa.jpg",
              "182169179_jpg.rf.4b645d23a7fc18ba8cb189121f9defea.jpg"],
}

# §8#1: gambar transposed, arah rotasi terverifikasi (scripts/check_rotation.py,
# dikonfirmasi teks pada valid/182169109 = "Chicken Waffle" tegak di rot270)
ROTATION_FIX = {
    ("train", "182169121_jpg.rf.eb0ccb71d0ea38adf1e38bf9c02a4a15.jpg"): cv2.ROTATE_90_CLOCKWISE,
    ("valid", "182169109_jpg.rf.0531e60eb6e404d9633ed0c23a381fe7.jpg"): cv2.ROTATE_90_COUNTERCLOCKWISE,
    ("valid", "182169108_jpg.rf.205d9f1ed8f888006b8f97fc29eb8790.jpg"): cv2.ROTATE_90_CLOCKWISE,
}
# catatan: train/182169107 juga transposed, tetapi dibuang sebagai leakage.

# §8#4: stuff vs things (heuristik panoptic, sama dengan EDA)
STUFF = {"wall", "wall-brick", "wall-stone", "wall-tile", "wall-wood",
         "wall-other-merged", "floor-other-merged", "floor-wood",
         "ceiling-merged", "rug-merged", "door-stuff", "mirror-stuff",
         "window-other", "window-blind", "curtain", "stairs", "tree-merged",
         "counter"}

# §8#5: ambang duplikat anotasi
DUP_BBOX_IOU = 0.85
DUP_MASK_IOU = 0.70

LOG = {"categories_dropped": [], "categories_merged": [],
       "leakage_removed": [], "images_rotated": [], "annotations_deduped": [],
       "images_emptied": [], "annotations_reclassed": [], "bbox_fixed": [],
       "areas_recomputed": 0, "coords_clipped": 0}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ geometri
def polygon_area(xy):
    if len(xy) < 3:
        return 0.0
    x, y = xy[:, 0], xy[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def rasterize(seg, w, h):
    m = np.zeros((h, w), np.uint8)
    for poly in seg:
        pts = np.round(np.array(poly, np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def bbox_iou(a, b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union else 0.0


# ------------------------------------------------------------- 1. kategorisasi
log("=" * 78)
log("1. KEBIJAKAN KATEGORI")
log("=" * 78)

raw_cats = json.load(open(glob.glob(os.path.join(ROOT, "train", "*.json"))[0],
                          encoding="utf-8"))["categories"]
name_by_id = {c["id"]: c["name"] for c in raw_cats}
supercategory = {c["name"]: c.get("supercategory", "") for c in raw_cats}

all_names = sorted(set(name_by_id.values()))
final_names = []
for n in all_names:
    if n in DROP_CATEGORIES:
        LOG["categories_dropped"].append(n)
        continue
    final_names.append(MERGE_CATEGORIES.get(n, n))
final_names = sorted(set(final_names))
# 'wallet'/'trash-bin' disebut eksplisit; pastikan tidak ada duplikat nama
assert len(final_names) == len(set(final_names))

name_to_new_id = {n: i + 1 for i, n in enumerate(final_names)}
old_to_new = {}
for oid, n in name_by_id.items():
    if n in DROP_CATEGORIES:
        old_to_new[oid] = None
    else:
        new_n = MERGE_CATEGORIES.get(n, n)
        old_to_new[oid] = name_to_new_id[new_n]
        if new_n != n:
            LOG["categories_merged"].append({"from": n, "to": new_n})

log(f"  kategori mentah : {len(all_names)}")
log(f"  dibuang         : {LOG['categories_dropped']}")
log(f"  digabung        : {LOG['categories_merged']}")
log(f"  kategori final  : {len(final_names)}")

cat_out = [{"id": name_to_new_id[n], "name": n,
            "supercategory": supercategory.get(n, ""),
            "isthing": 0 if n in STUFF else 1} for n in final_names]
n_stuff_out = sum(1 for c in cat_out if c["isthing"] == 0)
log(f"  stuff/things    : {n_stuff_out} stuff / {len(cat_out) - n_stuff_out} things")

# ----------------------------------------------------------------- 2. proses
log("\n" + "=" * 78)
log("2. PEMROSESAN PER SPLIT")
log("=" * 78)

global_img_offset = {"train": 0, "valid": 1_000_000, "test": 2_000_000}
manifest_rows = []
out_by_split = {}

for split in SPLITS:
    src = json.load(open(glob.glob(os.path.join(ROOT, split, "*.json"))[0], encoding="utf-8"))
    im_by_id = {im["id"]: im for im in src["images"]}
    im_folder = os.path.join(ROOT, split)
    drop_set = set(LEAKAGE_DROP.get(split, []))
    log(f"\n[{split}] masuk: {len(src['images'])} gambar, {len(src['annotations'])} anotasi")

    # --- 2a. buang gambar leakage
    kept_images = []
    for im in src["images"]:
        if im["file_name"] in drop_set:
            LOG["leakage_removed"].append(f"{split}/{im['file_name']}")
            continue
        kept_images.append(im)
    log(f"  leakage dibuang      : {len(src['images']) - len(kept_images)}")

    # --- 2b. filter anotasi kategori + dedup
    anns = []
    kept_ids = {im["id"] for im in kept_images}
    for a in src["annotations"]:
        if a["image_id"] not in kept_ids:
            continue
        new_cat = old_to_new.get(a["category_id"])
        if new_cat is None:
            continue
        a = dict(a)
        a["category_id"] = new_cat
        anns.append(a)
    n_after_cat = len(anns)
    log(f"  anotasi setelah filter kategori: {n_after_cat}")

    # --- 2c. perbaiki rotasi gambar transposed
    im_meta = {}
    for im in kept_images:
        key = (split, im["file_name"])
        src_path = os.path.join(im_folder, im["file_name"])
        w, h = im["width"], im["height"]
        rel_path = f"{split}/{im['file_name']}"
        if key in ROTATION_FIX:
            code = ROTATION_FIX[key]
            img = cv2.imread(src_path)
            rot = cv2.rotate(img, code)
            dst = os.path.join(FIX_DIR, im["file_name"])
            cv2.imwrite(dst, rot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            w, h = rot.shape[1], rot.shape[0]
            rel_path = f"data/processed/images_fixed/{im['file_name']}"
            LOG["images_rotated"].append(
                {"split": split, "file_name": im["file_name"],
                 "direction": "rot90_cw" if code == cv2.ROTATE_90_CLOCKWISE else "rot90_ccw",
                 "from": f"{im['width']}x{im['height']}", "to": f"{w}x{h}"})
        im_meta[im["id"]] = dict(rel_path=rel_path, w=w, h=h)

    # --- 2d. min-max koreksi geometri: klip koordinat, bbox ketat, area mask
    for a in anns:
        meta = im_meta[a["image_id"]]
        W, H = meta["w"], meta["h"]
        polys = []
        for poly in a["segmentation"]:
            arr = np.array(poly, dtype=float).reshape(-1, 2)
            clipped = np.clip(arr, [0.0, 0.0], [W - 0.01, H - 0.01])
            if not np.allclose(arr, clipped):
                LOG["coords_clipped"] += 1
            polys.append([round(float(v), 2) for v in clipped.reshape(-1)])
        a["segmentation"] = polys

        xs = np.concatenate([np.array(p, float).reshape(-1, 2)[:, 0] for p in polys])
        ys = np.concatenate([np.array(p, float).reshape(-1, 2)[:, 1] for p in polys])
        tight = [round(float(xs.min()), 2), round(float(ys.min()), 2),
                 round(float(xs.max() - xs.min()), 2), round(float(ys.max() - ys.min()), 2)]
        if (abs(tight[0] - a["bbox"][0]) > 2 or abs(tight[1] - a["bbox"][1]) > 2
                or abs(tight[2] - a["bbox"][2]) > 2 or abs(tight[3] - a["bbox"][3]) > 2):
            LOG["bbox_fixed"].append({"split": split, "image": a["image_id"],
                                      "old": [round(v, 1) for v in a["bbox"]],
                                      "new": tight})
        a["bbox"] = tight
        a["area"] = round(sum(polygon_area(np.array(p, float).reshape(-1, 2))
                              for p in polys), 2)
        LOG["areas_recomputed"] += 1

    # --- 2e. dedup anotasi (kategori sama, bbox IoU tinggi, mask hampir identik)
    by_img = defaultdict(list)
    for a in anns:
        by_img[a["image_id"]].append(a)
    kept = []
    for iid, group in by_img.items():
        meta = im_meta[iid]
        W, H = meta["w"], meta["h"]
        if len(group) == 1:
            kept.extend(group)
            continue
        masks = [None] * len(group)
        removed = set()
        order = sorted(range(len(group)), key=lambda k: -group[k]["area"])
        for ii in range(len(order)):
            i = order[ii]
            if i in removed:
                continue
            for jj in range(ii + 1, len(order)):
                j = order[jj]
                if j in removed or group[i]["category_id"] != group[j]["category_id"]:
                    continue
                if bbox_iou(group[i]["bbox"], group[j]["bbox"]) <= DUP_BBOX_IOU:
                    continue
                if masks[i] is None:
                    masks[i] = rasterize(group[i]["segmentation"], W, H)
                if masks[j] is None:
                    masks[j] = rasterize(group[j]["segmentation"], W, H)
                if mask_iou(masks[i], masks[j]) > DUP_MASK_IOU:
                    removed.add(j)
                    LOG["annotations_deduped"].append(
                        {"split": split, "file_name": im_by_id[iid]["file_name"],
                         "category": final_names[group[i]["category_id"] - 1],
                         "bbox_iou": round(bbox_iou(group[i]["bbox"], group[j]["bbox"]), 3),
                         "mask_iou": round(mask_iou(masks[i], masks[j]), 3)})
        kept.extend(group[k] for k in range(len(group)) if k not in removed)
    anns = kept
    log(f"  anotasi setelah dedup: {len(anns)} (buang {n_after_cat - len(anns)})")

    # --- 2f. buang gambar tanpa anotasi
    has_ann = {a["image_id"] for a in anns}
    final_images = []
    for im in kept_images:
        if im["id"] not in has_ann:
            LOG["images_emptied"].append(f"{split}/{im['file_name']}")
            continue
        final_images.append(im)
    log(f"  gambar tanpa anotasi dibuang: {len(kept_images) - len(final_images)}")
    log(f"  hasil: {len(final_images)} gambar, {len(anns)} anotasi")

    # --- 2g. reindex & tulis
    id_map = {}
    images_out = []
    for k, im in enumerate(final_images):
        new_id = global_img_offset[split] + k + 1
        id_map[im["id"]] = new_id
        meta = im_meta[im["id"]]
        images_out.append({
            "id": new_id, "file_name": meta["rel_path"],
            "width": meta["w"], "height": meta["h"],
            "split": split,
            "original_file_name": im["file_name"],
            "license": im.get("license", 0),
            "date_captured": im.get("date_captured", ""),
        })
        n_ann = sum(1 for a in anns if a["image_id"] == im["id"])
        manifest_rows.append({
            "split": split, "image_id": new_id, "file_name": meta["rel_path"],
            "original_file_name": im["file_name"],
            "width": meta["w"], "height": meta["h"],
            "megapixels": round(meta["w"] * meta["h"] / 1e6, 3),
            "n_annotations": n_ann,
            "n_things": sum(1 for a in anns if a["image_id"] == im["id"]
                            and cat_out[a["category_id"] - 1]["isthing"] == 1),
            "rotated": 1 if (split, im["file_name"]) in ROTATION_FIX else 0,
        })

    anns_out = []
    for k, a in enumerate(sorted(anns, key=lambda x: (x["image_id"], x["id"]))):
        anns_out.append({
            "id": k + 1, "image_id": id_map[a["image_id"]],
            "category_id": a["category_id"], "bbox": a["bbox"],
            "area": a["area"], "iscrowd": int(a.get("iscrowd", 0)),
            "segmentation": a["segmentation"],
        })

    coco = {"info": {**src.get("info", {}),
                      "preprocessing": "cleaned per reports/eda/EDA_REPORT.md §8"},
            "licenses": src.get("licenses", []),
            "images": images_out, "annotations": anns_out, "categories": cat_out}
    p = os.path.join(OUT_DIR, f"instances_{split}.json")
    json.dump(coco, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    out_by_split[split] = coco
    log(f"  -> {os.path.relpath(p, ROOT)}")

json.dump(cat_out, open(os.path.join(OUT_DIR, "categories.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
pd.DataFrame(manifest_rows).to_csv(os.path.join(OUT_DIR, "manifest.csv"),
                                   index=False, encoding="utf-8")

# ------------------------------------------------------- 3. kebijakan resize
resize_policy = """# Kebijakan resize (dibuat oleh scripts/preprocess.py)
# Tujuan: menyamakan skala input tanpa mengubah variable eksperimen (pretraining).
target_long_side: 640
keep_aspect_ratio: true
padding: letterbox      # pad nilai 114 (YOLO convention) / mean ImageNet untuk mmdet
upscale_limit: 2.0      # gambar test terkecil 149 px -> maksimal 2x, hindari upscale ekstrem
min_object_px: 4        # objek < 4 px di sisi terpanjang setelah resize -> tandai ignore
train_augmentation_scale: [0.5, 2.0]
eval_resize: letterbox_640
notes: |
  split train/valid sudah beresolusi 640x480 (90-92%), split test tidak
  (0% pada 640x480, rentang 149 px - 10.3 MP). Semua model dievaluasi dengan
  letterbox ke 640 px sisi terpanjang agar mAP sebanding antar-gambar.
  Jangan resize fisik dataset; lakukan di dataloader agar reproducible.
"""
open(os.path.join(OUT_DIR, "resize_policy.yaml"), "w", encoding="utf-8").write(resize_policy)

json.dump(LOG, open(os.path.join(OUT_DIR, "change_log.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

# ------------------------------------------------------------- 4. ringkasan
log("\n" + "=" * 78)
log("3. RINGKASAN PERUBAHAN")
log("=" * 78)
log(f"  kategori     : {len(all_names)} -> {len(final_names)}")
log(f"  gambar leakage dibuang  : {len(LOG['leakage_removed'])}")
log(f"  gambar dirotasi         : {len(LOG['images_rotated'])}")
log(f"  gambar jadi kosong      : {len(LOG['images_emptied'])}")
log(f"  anotasi dedup           : {len(LOG['annotations_deduped'])}")
log(f"  bbox dikoreksi          : {len(LOG['bbox_fixed'])}")
log(f"  koordinat diklip        : {LOG['coords_clipped']}")
log(f"  area dihitung ulang     : {LOG['areas_recomputed']}")

tot_img = sum(len(v["images"]) for v in out_by_split.values())
tot_ann = sum(len(v["annotations"]) for v in out_by_split.values())
log(f"\n  TOTAL AKHIR: {tot_img} gambar, {tot_ann} anotasi, {len(cat_out)} kategori")
log(f"  -> {os.path.relpath(OUT_DIR, ROOT)}")
