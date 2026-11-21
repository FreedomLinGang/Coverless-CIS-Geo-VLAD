"""VOC XML loading and simple VOC-style AP@0.5 evaluation."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import VOC_ANN, VOC_CLASSES, VOC_JPEG


def list_voc_images(max_images: int, seed: int) -> list[Path]:
    paths = sorted(VOC_JPEG.glob("*.jpg"))
    # keep only those with annotation
    paths = [p for p in paths if (VOC_ANN / f"{p.stem}.xml").exists()]
    if max_images and len(paths) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=max_images, replace=False)
        paths = [paths[i] for i in sorted(idx.tolist())]
    return paths


def list_coco_images(folder: Path, max_images: int, seed: int) -> list[Path]:
    paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if max_images and len(paths) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=max_images, replace=False)
        paths = [paths[i] for i in sorted(idx.tolist())]
    return paths


def parse_voc_gt(xml_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in VOC_CLASSES:
            continue
        bnd = obj.find("bndbox")
        box = [
            float(bnd.findtext("xmin")),
            float(bnd.findtext("ymin")),
            float(bnd.findtext("xmax")),
            float(bnd.findtext("ymax")),
        ]
        objs.append({"label": name, "box": box})
    return objs


def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def eval_voc_map(
    preds_by_image: dict[str, list[dict]],
    image_paths: list[Path],
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Compute per-class AP@IoU and mAP."""
    gts: dict[str, list[dict]] = {}
    npos = defaultdict(int)
    for p in image_paths:
        gt = parse_voc_gt(VOC_ANN / f"{p.stem}.xml")
        gts[str(p)] = gt
        for o in gt:
            npos[o["label"]] += 1

    aps = {}
    for cls in VOC_CLASSES:
        # gather detections
        dets = []
        for p in image_paths:
            for d in preds_by_image.get(str(p), []):
                if d["label"] == cls:
                    dets.append((str(p), d["score"], d["box"]))
        dets.sort(key=lambda x: -x[1])
        if npos[cls] == 0:
            continue
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        matched = defaultdict(set)
        for i, (img_id, score, box) in enumerate(dets):
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
