"""After coco_train cache is ready: build train+val schemes and update report."""
from __future__ import annotations

import json
from pathlib import Path

from geo_refine.adaptive_scheme import build_schemes
from geo_refine.config import CACHE, RESULTS
from geo_refine.geo_subclass_study import merge_caches, write_dataset_md


def main():
    train_caches = sorted(CACHE.glob("coco_train_yolov8m_n*_c0.5.json"))
    val = CACHE / "coco_val_yolov8m_n5000_c0.5.json"
    if not train_caches or not val.exists():
        raise SystemExit("missing coco train/val caches")
    c_train = max(train_caches, key=lambda p: p.stat().st_size)
    preds = json.loads(c_train.read_text(encoding="utf-8"))
    print(f"train cache {c_train.name}: images={len(preds)}")
    # expect ~118287
    dets = merge_caches([c_train, val])
    sch = build_schemes(dets)

    out_path = RESULTS / "geo_subclass_yolov8m.json"
    out = {}
    if out_path.exists():
        out = json.loads(out_path.read_text(encoding="utf-8"))
    out["detector"] = "yolov8m"
    out["conf"] = 0.5
    out["COCO2017_trainval"] = sch
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # rebuild full md from json parts
    lines = [
        "# Geometric Subclass Schemes (YOLOv8m, conf=0.5)",
        "",
        "## Method",
        "",
        "1. **Diag angle** `atan(h/w)` in (0,90 deg]: box shape (reference 4 bins x 22.5 deg).",
        "2. **Spatial angle** image-center to box-center in [0,360): location (reference 8 bins x 45 deg).",
        "3. **Dist ratio** `|c_box-c_img|/(0.5*image_diagonal)`: near/far (class-wise quantiles).",
        "",
        "Per class: grid-search (D,S,R) with product <= count/min_count; maximize usable subclasses "
        "with low CV and low OTHER; merge sparse into `|OTHER`.",
        "Uniform 4x8x4 is a sparse reference scheme only.",
        "",
        "Datasets: VOC2007 train+test, VOC2012 full, COCO2017 train+val.",
        "",
    ]
    if "VOC2007" in out:
        write_dataset_md("VOC2007 (train+test)", out["VOC2007"], lines)
    if "VOC2012" in out:
        write_dataset_md("VOC2012 (full)", out["VOC2012"], lines)
    write_dataset_md("COCO2017 (train+val)", sch, lines)
    lines += [
        "## Recommended pattern (for stego codebook)",
        "",
        "- Prefer **adaptive** schemes; avoid fixed 4x8x4.",
        "- Frequent classes: often spatial+dist (e.g. 1x8x4 or 2x4x2), sometimes +diag.",
        "- Mid classes: 1-2 dims; rare: no split.",
        "- Distance uses **quantiles** for balance.",
        "",
    ]
    md = RESULTS / "geo_subclass_yolov8m.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(
        "COCO dets",
        sch["total_detections"],
        "merged",
        sch["adaptive_summary"]["n_subclasses_merged"],
        "min_count",
        sch["min_count"],
    )
    print("Wrote", md)


if __name__ == "__main__":
    main()
