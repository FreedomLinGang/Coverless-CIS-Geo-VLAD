"""Full-dataset detector comparison: mAP + detected vs GT counts.

Models: Faster R-CNN, YOLOv8m, YOLOv8x (higher capacity).
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from geo_refine.config import CACHE, RESULTS, VOC_ANN, VOC_CLASSES
from geo_refine.detectors import build_detector
from geo_refine.voc_eval import eval_voc_map, iou, list_voc_images, parse_voc_gt


def gt_counts(image_paths: list[Path]) -> Counter:
    c = Counter()
    for p in image_paths:
        for o in parse_voc_gt(VOC_ANN / f"{p.stem}.xml"):
            c[o["label"]] += 1
    return c


def matched_tp_counts(
    preds_by_image: dict[str, list[dict]],
    image_paths: list[Path],
    iou_thresh: float = 0.5,
) -> tuple[Counter, Counter, Counter]:
    """Return (tp_per_class, fp_per_class, fn_per_class) at IoU thresh (greedy match)."""
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for p in image_paths:
        gts = parse_voc_gt(VOC_ANN / f"{p.stem}.xml")
        preds = sorted(preds_by_image.get(str(p), []), key=lambda d: -d["score"])
        used = set()
        for d in preds:
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gts):
                if j in used or g["label"] != d["label"]:
                    continue
                v = iou(d["box"], g["box"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_thresh:
                tp[d["label"]] += 1
                used.add(best_j)
            else:
                fp[d["label"]] += 1
        for j, g in enumerate(gts):
            if j not in used:
                fn[g["label"]] += 1
    return tp, fp, fn


def run_one(name: str, paths: list[Path], conf: float, device: str) -> dict:
    cache = CACHE / f"voc_full_{name}_n{len(paths)}_c{conf}.json"
    if cache.exists():
        preds = json.loads(cache.read_text(encoding="utf-8"))
    else:
        det = build_detector(name, device=device)
        preds = {}
        t0 = time.time()
        for p in tqdm(paths, desc=f"detect[{name}]"):
            try:
                preds[str(p)] = det.detect(p, conf=conf, voc_mode=True)
            except Exception as e:
                print(f"[warn] {p}: {e}")
                preds[str(p)] = []
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(preds), encoding="utf-8")
        print(f"{name} detect seconds={time.time()-t0:.1f}")

    t0 = time.time()
    metrics = eval_voc_map(preds, paths, iou_thresh=0.5)
    gt = gt_counts(paths)
    det_c = Counter(d["label"] for v in preds.values() for d in v)
    tp, fp, fn = matched_tp_counts(preds, paths, 0.5)
    rows = {}
    for cls in VOC_CLASSES:
        g = gt[cls]
        d = det_c[cls]
        rows[cls] = {
            "gt": g,
            "detected": d,
            "detected_minus_gt": d - g,
            "recall_proxy": (tp[cls] / g) if g else None,  # TP / GT
            "precision_proxy": (tp[cls] / d) if d else None,  # TP / Det
            "tp": tp[cls],
            "fp": fp[cls],
            "fn": fn[cls],
            "ap50": metrics["per_class_ap"].get(cls),
        }
    return {
        "detector": name,
        "n_images": len(paths),
        "mAP50": metrics["mAP50"],
        "total_gt": int(sum(gt.values())),
        "total_detected": int(sum(det_c.values())),
        "total_tp": int(sum(tp.values())),
        "total_fp": int(sum(fp.values())),
        "total_fn": int(sum(fn.values())),
        "eval_seconds": time.time() - t0,
        "per_class": rows,
    }


def write_md(results: list[dict], path: Path):
    lines = []
    lines.append("# Full VOC2012 Detector Comparison\n")
    lines.append("## Why Faster R-CNN looked best (previous 500-img run)\n")
    lines.append(
        "- All models were **COCO-pretrained**, then evaluated on VOC with name mapping "
        "(airplane→aeroplane etc.), not VOC-finetuned.\n"
        "- FRCNN often outputs **more boxes** at the same conf; YOLO is more conservative → "
        "higher recall can raise mAP even if not the SOTA detector family.\n"
        "- Gap was small (0.732 vs 0.728); **not** proof FRCNN is universally best. "
        "Larger YOLO (v8x) / RT-DETR usually beat FRCNN-ResNet50 on COCO benchmarks.\n"
    )
    lines.append("## Summary (all annotated VOC2012 images)\n")
    lines.append("| Detector | mAP@0.5 | GT objs | Detected | TP | FP | FN | Det/GT |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        ratio = r["total_detected"] / max(1, r["total_gt"])
        lines.append(
            f"| {r['detector']} | {r['mAP50']:.4f} | {r['total_gt']} | {r['total_detected']} | "
            f"{r['total_tp']} | {r['total_fp']} | {r['total_fn']} | {ratio:.3f} |"
        )
    best = max(results, key=lambda x: x["mAP50"])
    lines.append(f"\n**Best mAP:** `{best['detector']}` = {best['mAP50']:.4f}\n")

    for r in results:
        lines.append(f"## Per-class: `{r['detector']}`\n")
        lines.append("| Class | GT | Detected | Det-GT | TP | FP | FN | Recall | Precision | AP50 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cls, row in r["per_class"].items():
            rec = row["recall_proxy"]
            prec = row["precision_proxy"]
            ap = row["ap50"]
            lines.append(
                f"| {cls} | {row['gt']} | {row['detected']} | {row['detected_minus_gt']:+d} | "
                f"{row['tp']} | {row['fp']} | {row['fn']} | "
                f"{(rec if rec is not None else 0):.3f} | "
                f"{(prec if prec is not None else 0):.3f} | "
                f"{(ap if ap is not None else 0):.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # all images with annotation
    paths = list_voc_images(max_images=0, seed=42)  # 0 => need fix: currently max_images falsy uses all?
    # list_voc_images: if max_images and len > max → subsample; max_images=0 is falsy → all
    print(f"VOC images with ann: {len(paths)}")
    detectors = ["fasterrcnn", "yolov8m", "yolov8x"]
    conf = 0.5
    device = "cuda"
    results = []
    for name in detectors:
        print(f"==== {name} ====")
        try:
            results.append(run_one(name, paths, conf, device))
            print(
                f"{name}: mAP={results[-1]['mAP50']:.4f} "
                f"det={results[-1]['total_detected']} gt={results[-1]['total_gt']}"
            )
        except Exception as e:
            print(f"[fail] {name}: {e}")
            results.append({"detector": name, "error": str(e)})

    out_json = RESULTS / "voc_full_detector_compare.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = [r for r in results if "mAP50" in r]
    if ok:
        write_md(ok, RESULTS / "voc_full_detector_compare.md")
        print("Wrote", RESULTS / "voc_full_detector_compare.md")


if __name__ == "__main__":
    main()
