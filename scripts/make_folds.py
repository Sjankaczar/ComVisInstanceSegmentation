"""Buat split 5-fold stratified multi-label dari train+valid yang sudah dibersihkan.

Output: splits/folds.json  +  reports/preprocess/fold_coverage.csv
"""
import os, sys, json, random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "processed")
SPLIT_DIR = os.path.join(ROOT, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

K = 5
SEED = 42

cats = {c["id"]: c for c in json.load(open(os.path.join(OUT, "categories.json"), encoding="utf-8"))}
cat_name = {cid: c["name"] for cid, c in cats.items()}

images, img_labels = [], {}
for split in ["train", "valid"]:
    d = json.load(open(os.path.join(OUT, f"instances_{split}.json"), encoding="utf-8"))
    for im in d["images"]:
        key = im["id"]
        images.append({"id": key, "split": split, "file_name": im["file_name"],
                       "original_file_name": im["original_file_name"],
                       "width": im["width"], "height": im["height"]})
        img_labels[key] = set()
    for a in d["annotations"]:
        img_labels[a["image_id"]].add(cat_name[a["category_id"]])

print(f"gambar: {len(images)} | kelas: {len(cats)}")
print(f"kelas per gambar: mean={np.mean([len(v) for v in img_labels.values()]):.1f}")

# ------------------------------- iterative stratification (Sechidis et al.)
label_count = Counter()
for v in img_labels.values():
    for lab in v:
        label_count[lab] += 1

W_SIZE = 2.0  # bobot penyeimbang ukuran fold; dipilih lewat scripts/tune_folds.py
order = sorted(label_count, key=lambda l: label_count[l])  # langka lebih dulu
folds = [set() for _ in range(K)]
fold_label_count = [Counter() for _ in range(K)]
assigned = set()
target_lab = {l: label_count[l] / K for l in label_count}
target_size = len(images) / K

for lab in order:
    idxs = [im["id"] for im in images
            if lab in img_labels[im["id"]] and im["id"] not in assigned]
    # gambar dengan label langka tambahan didahulukan
    idxs.sort(key=lambda i: len(img_labels[i]), reverse=True)
    for i in idxs:
        f = min(range(K), key=lambda k: (fold_label_count[k][lab] / target_lab[lab])
                + W_SIZE * (len(folds[k]) / target_size))
        folds[f].add(i)
        assigned.add(i)
        # hitung SEMUA label gambar ini, bukan hanya label yang sedang diproses
        for l2 in img_labels[i]:
            fold_label_count[f][l2] += 1

unassigned = [im["id"] for im in images if im["id"] not in assigned]
for i in unassigned:
    f = min(range(K), key=lambda k: len(folds[k]))
    folds[f].add(i)
    for lab in img_labels[i]:
        fold_label_count[f][lab] += 1

print(f"\nukuran fold: {[len(f) for f in folds]}")
print(f"gambar tanpa label: {len(unassigned)}")

# ------------------------------------------------------------ verifikasi
rows = []
for lab in sorted(label_count):
    row = {"class": lab, "total": label_count[lab]}
    for k in range(K):
        row[f"f{k}_val"] = fold_label_count[k][lab]
    row["fold_hilang"] = sum(1 for k in range(K) if fold_label_count[k][lab] == 0)
    rows.append(row)
cov = pd.DataFrame(rows)
cov.to_csv(os.path.join(ROOT, "reports", "preprocess", "fold_coverage.csv"), index=False)

print(f"\nkelas hilang di N fold (dari 5):")
print(cov.fold_hilang.value_counts().sort_index().to_string())
print(f"kelas muncul di semua 5 fold: {(cov.fold_hilang == 0).sum()} / {len(cov)}")
print(f"\nkelas yang hilang di >=2 fold:")
sub = cov[cov.fold_hilang >= 2]
if len(sub):
    print(sub[["class", "total"] + [f"f{k}_val" for k in range(K)]].to_string(index=False))
else:
    print("  (tidak ada)")

# --------------------------------------------------------------- simpan
out = {"seed": SEED, "k": K, "note": "stratified multi-label over cleaned train+valid",
       "w_size": W_SIZE,
       "classes_absent_in_trainval": sorted(set(cat_name.values()) - set(label_count)),
       "folds": []}
for k in range(K):
    val_ids = sorted(folds[k])
    train_ids = sorted(set(im["id"] for im in images) - folds[k])
    out["folds"].append({
        "fold": k,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
    })
json.dump(out, open(os.path.join(SPLIT_DIR, "folds.json"), "w", encoding="utf-8"), indent=2)

# daftar file per fold, praktis untuk training
listdir = os.path.join(SPLIT_DIR, "lists")
os.makedirs(listdir, exist_ok=True)
by_id = {im["id"]: im for im in images}
for k in range(K):
    for part in ["train", "val"]:
        ids = out["folds"][k][f"{part}_ids"]
        with open(os.path.join(listdir, f"f{k}_{part}.txt"), "w", encoding="utf-8") as f:
            for i in ids:
                f.write(by_id[i]["file_name"] + "\n")
print(f"\nsaved: splits/folds.json, splits/lists/f{k}_{{train,val}}.txt")
