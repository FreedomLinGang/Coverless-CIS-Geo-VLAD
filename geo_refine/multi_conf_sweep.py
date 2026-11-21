"""Multi-dataset detector compare + confidence sweep.

Datasets:
  - VOC2007 (train + test)
  - VOC2012 (full)
  - COCO2017 val (instances GT; train too large for full mAP pass)

Detectors: fasterrcnn, yolov8m, yolov8x
Conf: 0.5 detect once, then filter for 0.5/0.6/0.7/0.8/0.9
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from geo_refine.config import (
    CACHE,
    COCO_ANN_VAL,
    COCO_VAL,
    RESULTS,
    VOC2007_TEST,
    VOC2007_TRAIN,
    VOC2012_ROOT,
    VOC_CLASSES,
)
from geo_refine.detectors import build_detector
from geo_refine.voc_eval import iou, parse_voc_gt, voc_ap
import numpy as np

DETECTORS = ["fasterrcnn", "yolov8m", "yolov8x"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def list_voc_split(root: Path) -> list[Path]:
    jpeg = root / "JPEGImages"
    ann = root / "Annotations"
    paths = sorted(jpeg.glob("*.jpg"))
    return [p for p in paths if (ann / f"{p.stem}.xml").exists()]


def load_voc_gts(roots: list[Path]) -> tuple[list[Path], dict[str, list[dict]]]:
    paths: list[Path] = []
    gts: dict[str, list[dict]] = {}
    for root in roots:
        ann = root / "Annotations"
        for p in list_voc_split(root):
            paths.append(p)
            gts[str(p)] = parse_voc_gt(ann / f"{p.stem}.xml")
    return paths, gts


def load_coco_val_gts(
    img_dir: Path = COCO_VAL,
    ann_path: Path = COCO_ANN_VAL,
) -> tuple[list[Path], dict[str, list[dict]], list[str]]:
    """Load COCO val images + GT boxes (category name as label)."""
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    classes = sorted(id2name.values())
    img_by_id = {im["id"]: im for im in data["images"]}
    gts: dict[str, list[dict]] = defaultdict(list)
    for a in data["annotations"]:
        if a.get("iscrowd", 0):
            continue
        im = img_by_id[a["image_id"]]
        path = img_dir / im["file_name"]
        x, y, w, h = a["bbox"]  # xywh
        gts[str(path)].append(
            {
                "label": id2name[a["category_id"]],
                "box": [float(x), float(y), float(x + w), float(y + h)],
            }
        )
    paths = sorted(Path(p) for p in gts.keys() if Path(p).exists())
    # also include images with zero objects? skip; eval only images that exist
    # add empty images that exist in folder and in json
    present = {str(p) for p in paths}
    for im in data["images"]:
        p = img_dir / im["file_name"]
        if p.exists() and str(p) not in present:
            gts[str(p)] = []
            paths.append(p)
    paths = sorted(paths)
    return paths, dict(gts), classes


def matched_tp_fp_fn(
    preds_by_image: dict[str, list[dict]],
    gts: dict[str, list[dict]],
    image_paths: list[Path],
    iou_thresh: float = 0.5,
) -> tuple[Counter, Counter, Counter]:
    tp, fp, fn = Counter(), Counter(), Counter()
    for p in image_paths:
        key = str(p)
        gt_list = gts.get(key, [])
        preds = sorted(preds_by_image.get(key, []), key=lambda d: -d["score"])
        used = set()
        for d in preds:
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gt_list):
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
        for j, g in enumerate(gt_list):
            if j not in used:
                fn[g["label"]] += 1
    return tp, fp, fn


def eval_map(
    preds_by_image: dict[str, list[dict]],
    gts: dict[str, list[dict]],
    image_paths: list[Path],
    classes: list[str],
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    npos: dict[str, int] = defaultdict(int)
    for p in image_paths:
        for o in gts.get(str(p), []):
            npos[o["label"]] += 1

    aps = {}
    for cls in classes:
        if npos.get(cls, 0) == 0:
            continue
        dets = []
        for p in image_paths:
            for d in preds_by_image.get(str(p), []):
                if d["label"] == cls:
                    dets.append((str(p), d["score"], d["box"]))
        dets.sort(key=lambda x: -x[1])
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        matched: dict[str, set[int]] = defaultdict(set)
        for i, (img_id, _score, box) in enumerate(dets):
            gt_boxes = [g["box"] for g in gts[img_id] if g["label"] == cls]
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if j in matched[img_id]:
                    continue
                v = iou(box, gb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thresh and best_j >= 0:
                tp[i] = 1
                matched[img_id].add(best_j)
            else:
                fp[i] = 1
        if len(dets) == 0:
            aps[cls] = 0.0
            continue
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec = tp_c / float(npos[cls])
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        aps[cls] = voc_ap(rec, prec)
    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
    return {"per_class_ap": aps, "mAP50": mean_ap, "npos": dict(npos)}


def filter_preds(preds: dict, conf: float) -> dict:
    return {k: [d for d in v if float(d["score"]) >= conf] for k, v in preds.items()}


def run_detect(
    name: str,
    paths: list[Path],
    conf: float,
    device: str,
    voc_mode: bool,
    cache: Path,
) -> dict[str, list[dict]]:
    if cache.exists():
        print(f"  load cache {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    det = build_detector(name, device=device)
    preds: dict[str, list[dict]] = {}
    t0 = time.time()
    for p in tqdm(paths, desc=f"detect[{name}]"):
        try:
            preds[str(p)] = det.detect(p, conf=conf, voc_mode=voc_mode)
        except Exception as e:
            print(f"[warn] {p}: {e}")
            preds[str(p)] = []
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(preds), encoding="utf-8")
    print(f"  {name} detect seconds={time.time() - t0:.1f}")
    return preds


def summarize(
    name: str,
    conf: float,
    preds: dict,
    paths: list[Path],
    gts: dict[str, list[dict]],
    classes: list[str],
) -> dict:
    metrics = eval_map(preds, gts, paths, classes, 0.5)
    tp, fp, fn = matched_tp_fp_fn(preds, gts, paths, 0.5)
    det_c = Counter(d["label"] for v in preds.values() for d in v)
    gt_n = sum(len(gts.get(str(p), [])) for p in paths)
    det_n = int(sum(det_c.values()))
    tp_n = int(sum(tp.values()))
    fp_n = int(sum(fp.values()))
    fn_n = int(sum(fn.values()))
    return {
        "detector": name,
        "conf": conf,
        "mAP50": metrics["mAP50"],
        "n_images": len(paths),
        "total_gt": gt_n,
        "total_detected": det_n,
        "total_tp": tp_n,
        "total_fp": fp_n,
        "total_fn": fn_n,
        "det_over_gt": det_n / max(1, gt_n),
        "recall": tp_n / max(1, gt_n),
        "precision": tp_n / max(1, det_n),
    }


def write_md(all_rows: dict[str, list[dict]], path: Path):
    lines = [
        "# Multi-Dataset Confidence Threshold Sweep",
        "",
        "Detectors: Faster R-CNN, YOLOv8m, YOLOv8x. Conf filtered from ≥0.5 cache.",
        "Match: same class + IoU≥0.5.",
        "",
        "## Datasets",
        "",
        "- **VOC2007**: `D:\\datasets\\VOC2007` (train + test)",
        "- **VOC2012**: `D:\\datasets\\VOC2012` (full JPEGImages+Annotations)",
        "- **COCO2017-val**: `D:\\datasets\\coco\\val2017\\val2017` + instances_val2017.json",
        "  (train2017 has 118k images; used later for indexing; mAP reported on val)",
        "",
    ]
    for ds, rows in all_rows.items():
        lines.append(f"## {ds}")
        lines.append("")
        if not rows:
            lines.append("(no results)")
            lines.append("")
            continue
        gt = rows[0]["total_gt"]
        nimg = rows[0]["n_images"]
        lines.append(f"images={nimg}, GT objs={gt}")
        lines.append("")
        lines.append(
            "| Detector | Conf | mAP@0.5 | GT | Det | TP | FP | FN | Det/GT | Recall | Precision |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['detector']} | {r['conf']:.1f} | {r['mAP50']:.4f} | {r['total_gt']} | "
                f"{r['total_detected']} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} | "
                f"{r['det_over_gt']:.3f} | {r['recall']:.3f} | {r['precision']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def process_dataset(
    ds_name: str,
    paths: list[Path],
    gts: dict[str, list[dict]],
    classes: list[str],
    voc_mode: bool,
    device: str,
) -> list[dict]:
    print(f"\n======== {ds_name}: {len(paths)} images, GT={sum(len(gts.get(str(p), [])) for p in paths)} ========")
    rows: list[dict] = []
    for name in DETECTORS:
        cache = CACHE / f"{ds_name}_{name}_n{len(paths)}_c0.5.json"
        print(f"==== {name} ====")
        preds0 = run_detect(name, paths, 0.5, device, voc_mode, cache)
        for conf in THRESHOLDS:
            preds = filter_preds(preds0, conf)
            r = summarize(name, conf, preds, paths, gts, classes)
            rows.append(r)
            print(
                f"  conf={conf:.1f} mAP={r['mAP50']:.4f} det={r['total_detected']} "
                f"TP={r['total_tp']} FP={r['total_fp']} FN={r['total_fn']} "
                f"P={r['precision']:.3f} R={r['recall']:.3f}"
            )
    return rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    all_rows: dict[str, list[dict]] = {}

    # VOC2007
    p07, g07 = load_voc_gts([VOC2007_TRAIN, VOC2007_TEST])
    all_rows["VOC2007"] = process_dataset("voc2007", p07, g07, VOC_CLASSES, True, device)

    # VOC2012
    p12, g12 = load_voc_gts([VOC2012_ROOT])
    all_rows["VOC2012"] = process_dataset("voc2012", p12, g12, VOC_CLASSES, True, device)

    # COCO val
    pc, gc, coco_classes = load_coco_val_gts()
    all_rows["COCO2017_val"] = process_dataset("coco_val", pc, gc, coco_classes, False, device)

    out_json = RESULTS / "multi_conf_sweep.json"
    out_md = RESULTS / "multi_conf_sweep.md"
    out_json.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(all_rows, out_md)
    print("\nWrote", out_md)


if __name__ == "__main__":
    main()
