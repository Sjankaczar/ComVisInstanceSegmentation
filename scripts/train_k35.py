"""Training YOLO26-seg 35 kelas di lokal, meniru resep `long_run.py`.

Resep yang direproduksi (dari reports/SCORE_IMPROVEMENT.md + long_run.py):
  - AdamW eksplisit, lr0=5e-4, lrf=0.01, warmup_epochs=0
  - cls=1.0, copy_paste=0.3, close_mosaic=3
  - workers=0 (Windows), cache=disk, deterministic, seed=0
  - power plan High Performance selama training, Balanced sesudahnya
  - val dengan weights/best.pt di akhir, kurva mAP50-95 per epoch

Yang BERBEDA dari long_run.py:
  - dataset 35 kelas (24 things + 11 stuff), bukan fix22 (24 things)
  - batch menyesuaikan VRAM: angka terukur di RTX 4060 8 GB
  - mendukung arm scratch (weights=None dari yaml)
  - menulis metrics.json + config snapshot + log Tee

Contoh:
  python scripts/train_k35.py --model m --arm pretrained --epochs 24 --fold 0
  python scripts/train_k35.py --model m --arm scratch --epochs 48 --fold 0
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import win_patch  # noqa: F401,E402

RUNS = os.path.join(ROOT, "local_runs")
OUT = os.path.join(ROOT, "reports")

# batch aman terukur (RTX 4060 Laptop 8 GB, 35 kelas, 640px):
#   n bs16 -> 5.69 GB, 24.7 s/ep
#   s bs8  -> 5.16 GB, 33.2 s/ep
#   m bs4  -> 5.24 GB, 58.7 s/ep
#   l bs4  -> 6.19 GB, 57.7 s/ep
SAFE_BATCH = {"n": 16, "s": 8, "m": 4, "l": 4}


def set_power(high=True):
    """High Performance selama training, Balanced setelahnya."""
    guid = ("8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c" if high
            else "381b4222-f694-41f0-9685-ff5bb260df2e")
    import subprocess
    return subprocess.run(["powercfg", "/setactive", guid],
                          capture_output=True).returncode == 0


def curve(run_dir):
    """mAP50-95 mask per epoch dari results.csv."""
    p = os.path.join(run_dir, "results.csv")
    if not os.path.exists(p):
        return {}
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return {}
    key = next((k for k in rows[0] if k.strip() == "metrics/mAP50-95(M)"), None)
    if key is None:
        return {}
    out = {}
    for r in rows:
        try:
            out[int(float(r["epoch"]))] = round(float(r[key]), 4)
        except (ValueError, TypeError, KeyError):
            pass
    return out


def split_metric(v, name):
    """Metrik per-kelas untuk memisahkan things vs stuff."""
    names = v.names
    out = {}
    try:
        idx = getattr(v.seg, "ap_class_index", None)
        ap = getattr(v.seg, "ap", None)
        if idx is not None and ap is not None and len(ap):
            for k, ci in enumerate(list(idx)):
                if k < len(ap):
                    out[str(names[int(ci)])] = round(float(ap[k]), 4)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SAFE_BATCH))
    ap.add_argument("--arm", required=True, choices=["pretrained", "scratch"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--no-copy-paste", action="store_true")
    ap.add_argument("--resume-from", default=None,
                    help="path checkpoint untuk melanjutkan (bobot saja, "
                         "bukan resume optimizer)")
    args = ap.parse_args()

    from ultralytics import YOLO

    # arm scratch butuh ~2x epoch: bukti pilot, 10 epoch -> mAP 0.000
    epochs = args.epochs if args.epochs else (100 if args.arm == "pretrained"
                                              else 200)
    batch = args.batch or SAFE_BATCH[args.model]
    lr0 = args.lr0 if args.lr0 else (5.0e-4 if args.arm == "pretrained"
                                     else 1.0e-3)
    tag = args.tag or f"k35_{args.model}_{args.arm[0]}f{args.fold}"
    data = os.path.join(ROOT, "local_ds", f"k35_f{args.fold}", "data.yaml")
    if not os.path.exists(data):
        raise SystemExit(f"dataset tidak ada: {data}\n"
                         f"jalankan: python scripts/local_build_fold35.py")

    weights = (f"yolo26{args.model}-seg.pt" if args.arm == "pretrained"
               else f"yolo26{args.model}-seg.yaml")
    resume_note = None
    if args.resume_from:
        rp = args.resume_from
        if not os.path.isabs(rp):
            rp = os.path.join(ROOT, rp)
        if not os.path.exists(rp):
            raise SystemExit(f"checkpoint tidak ada: {rp}")
        import torch
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        # resume sejati butuh optimizer + ema + epoch >= 0; Ultralytics
        # menghapusnya saat training selesai normal.
        has_state = (ck.get("optimizer") is not None
                     and ck.get("ema") is not None
                     and int(ck.get("epoch", -1)) >= 0)
        resume_note = {"from": os.path.relpath(rp, ROOT),
                       "saved_epoch": ck.get("epoch"),
                       "has_optimizer": ck.get("optimizer") is not None,
                       "has_ema": ck.get("ema") is not None,
                       "true_resume_possible": has_state}
        if has_state:
            print("RESUME_SEJATI memungkinkan (optimizer+ema ada)")
        else:
            print("LANJUT_DARI_BOBOT: checkpoint tanpa optimizer/ema "
                  "(training lama selesai) — optimizer & scheduler dimulai "
                  "ulang, LR mengikuti lr0 baru")
        weights = rp

    run_dir = os.path.join(RUNS, tag)
    os.makedirs(run_dir, exist_ok=True)
    meta = json.load(open(os.path.join(ROOT, "local_ds",
                                       f"k35_f{args.fold}", "meta.json"),
                          encoding="utf-8"))

    cfg = {"tag": tag, "model": weights, "arm": args.arm, "fold": args.fold,
           "epochs": epochs, "batch": batch, "imgsz": args.imgsz,
           "lr0": lr0, "lrf": 0.01, "warmup_epochs": 0.0,
           "close_mosaic": 3, "optimizer": "AdamW", "cls": 1.0,
           "copy_paste": 0.0 if args.no_copy_paste else 0.3,
           "workers": 0, "cache": "disk", "seed": 0, "deterministic": True,
           "nc": meta["nc"], "n_things": meta["n_things"],
           "n_stuff": meta["n_stuff"], "data": data,
           "device": "cuda:0"}
    if resume_note:
        cfg["continued_from"] = resume_note
    json.dump(cfg, open(os.path.join(run_dir, "cfg.json"), "w"),
              indent=2, ensure_ascii=False)
    print("CFG " + json.dumps(cfg), flush=True)

    log_path = os.path.join(run_dir, "train.log")
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    set_power(True)
    log("power plan: High Performance")
    t0 = time.time()
    status, err = "complete", None
    try:
        m = YOLO(weights)
        nparam = sum(p.numel() for p in m.model.parameters())
        log(f"PARAMS {nparam:,} ({nparam/1e6:.1f}M) | arm={args.arm}")
        m.train(data=data, epochs=epochs, batch=batch, imgsz=args.imgsz,
                device=0, workers=0, seed=0, optimizer="AdamW", lr0=lr0,
                lrf=0.01, warmup_epochs=0.0, cls=1.0,
                copy_paste=0.0 if args.no_copy_paste else 0.3,
                close_mosaic=3,
                patience=args.patience if args.patience else epochs,
                project=RUNS, name=tag, exist_ok=True, plots=False,
                val=True, verbose=False, cache="disk", deterministic=True)
    except Exception as e:
        status = "failed"
        err = f"{type(e).__name__}: {str(e)[:300]}"
        traceback.print_exc()
        log("TRAIN GAGAL " + err)
    train_s = time.time() - t0

    c = curve(run_dir)
    res = {"tag": tag, "cfg": cfg, "status": status, "error": err,
           "train_seconds": round(train_s, 1),
           "s_per_epoch": round(train_s / max(epochs, 1), 1),
           "curve": c}

    best = os.path.join(run_dir, "weights", "best.pt")
    if status == "complete" and os.path.exists(best):
        try:
            v = YOLO(best).val(data=data, device=0, verbose=False,
                               project=run_dir, name="val", exist_ok=True)
            res.update({
                "val_map50_95_mask": round(float(v.seg.map), 4),
                "val_map50_mask": round(float(v.seg.map50), 4),
                "val_precision": round(float(v.seg.mp), 4),
                "val_recall": round(float(v.seg.mr), 4),
                "per_class_ap": split_metric(v, "seg"),
            })
            log(f"VAL mask mAP50-95={res['val_map50_95_mask']:.4f} "
                f"mAP50={res['val_map50_mask']:.4f}")
        except Exception as e:
            res["val_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            log("VAL GAGAL " + res["val_error"])
    elif status == "complete":
        res["error"] = "best.pt tidak ada setelah training"

    try:
        import torch
        res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        pass

    json.dump(res, open(os.path.join(run_dir, "metrics.json"), "w"),
              indent=2, ensure_ascii=False)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"),
              indent=2, ensure_ascii=False)

    if c:
        eps = sorted(c)
        log(f"kurva: ep{eps[0]}={c[eps[0]]:.4f} -> "
            f"ep{eps[len(eps)//2]}={c[eps[len(eps)//2]]:.4f} -> "
            f"ep{eps[-1]}={c[eps[-1]]:.4f}")
    log(f"SELESAI {status} | {res['train_seconds']/60:.1f} menit | "
        f"{res['s_per_epoch']:.1f} s/epoch")
    log("RESULT " + json.dumps({k: v for k, v in res.items()
                                if k != "per_class_ap"}))

    set_power(False)
    log("power plan: Balanced (dipulihkan)")
    logf.close()


if __name__ == "__main__":
    main()
