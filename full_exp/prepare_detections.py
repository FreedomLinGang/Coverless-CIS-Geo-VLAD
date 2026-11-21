# -*- coding: utf-8 -*-
"""Prepare YOLOv8m detections for full VOC2007 / VOC2012 / COCO train+val.

Reuses existing caches when possible; remaps outdated COCO absolute paths by basename.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from full_exp.config import (
    CACHE,
    COCO_TRAIN,
    COCO_VAL,
    CONF,
    DETECTOR,
    DEVICE,
    LEGACY_GEO_CACHE,
    VOC2007_TEST_JPEG,
    VOC2007_TRAIN_JPEG,
    VOC2012_JPEG,
)
from geo_refine.detectors import build_detector


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def remap_cache_by_basename(
    legacy_json: Path,
    image_dirs: list[Path],
    out_json: Path,
) -> dict[str, list]:
    """Rewrite detection keys to current absolute paths via filename match."""
    print(f"remap {legacy_json.name} → {out_json.name}")
    data = json.loads(legacy_json.read_text(encoding="utf-8"))
    name_to_path: dict[str, Path] = {}
    for d in image_dirs:
        for p in list_images(d):
            name_to_path[p.name] = p.resolve()
    out: dict[str, list] = {}
    miss = 0
    for old_key, dets in data.items():
        name = Path(old_key).name
        new_p = name_to_path.get(name)
        if new_p is None:
            miss += 1
            continue
        out[str(new_p)] = dets
    print(f"  remapped {len(out)}  missing_names={miss}  folder_imgs={len(name_to_path)}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out), encoding="utf-8")
    return out


def detect_folder_chunked(
    folder: Path,
    out_json: Path,
    *,
    conf: float = CONF,
    device: str = DEVICE,
    voc_mode: bool = False,
    chunk: int = 64,
    batch: int = 8,
) -> dict[str, list]:
    paths = list_images(folder)
    preds: dict[str, list] = {}
    if out_json.exists():
        preds = json.loads(out_json.read_text(encoding="utf-8"))
        print(f"resume {out_json.name}: {len(preds)}/{len(paths)}")
    todo = [p for p in paths if str(p.resolve()) not in preds and str(p) not in preds]
    # normalize keys to resolve()
    if preds and todo:
        # also skip if basename already present under another key
        have = {Path(k).name for k in preds}
        todo = [p for p in todo if p.name not in have]
    print(f"detect {folder}: todo={len(todo)} total={len(paths)}")
    if not todo:
        # rewrite keys to resolved paths if needed
        fixed = {str(Path(k).resolve()) if Path(k).exists() else k: v for k, v in preds.items()}
        out_json.write_text(json.dumps(fixed), encoding="utf-8")
        return fixed

    det = build_detector(DETECTOR, device=device)
    t0 = time.time()
    start = len(preds)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(0, len(todo), chunk), desc=f"det[{folder.name}]"):
        batch_paths = todo[i : i + chunk]
        with torch.inference_mode():
            results = det.model.predict(
                source=[str(p) for p in batch_paths],
                conf=conf,
                device=det.device,
                verbose=False,
                batch=batch,
                stream=False,
                imgsz=640,
            )
        for p, r in zip(batch_paths, results):
            preds[str(p.resolve())] = det._parse_result(r, voc_mode=voc_mode)
        del results
        torch.cuda.empty_cache()
        if (i // chunk) % 5 == 0:
            out_json.write_text(json.dumps(preds), encoding="utf-8")
            done = len(preds)
            rate = (done - start) / max(1e-6, time.time() - t0)
            print(f"  checkpoint {done}/{len(paths)}  {rate:.1f} img/s")
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    print(f"done {len(preds)} → {out_json}  ({time.time()-t0:.1f}s)")
    return preds


def merge_pred_dicts(dicts: list[dict[str, list]]) -> dict[str, list]:
    out: dict[str, list] = {}
    for d in dicts:
        out.update(d)
    return out


def prepare_all() -> dict[str, Path]:
    """Ensure detection caches exist; return mapping dataset_name → cache path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, Path] = {}

    # VOC2007 train+test
    voc07_out = CACHE / f"voc2007_{DETECTOR}_n9963_c{CONF}.json"
    if not voc07_out.exists():
        legacy = LEGACY_GEO_CACHE / f"voc2007_{DETECTOR}_n9963_c{CONF}.json"
        if legacy.exists():
            remap_cache_by_basename(
                legacy, [VOC2007_TRAIN_JPEG, VOC2007_TEST_JPEG], voc07_out
            )
        else:
            a = detect_folder_chunked(
                VOC2007_TRAIN_JPEG, CACHE / "_tmp_voc07_train.json", voc_mode=True
            )
            b = detect_folder_chunked(
                VOC2007_TEST_JPEG, CACHE / "_tmp_voc07_test.json", voc_mode=True
            )
            merged = merge_pred_dicts([a, b])
            voc07_out.write_text(json.dumps(merged), encoding="utf-8")
    out_paths["VOC2007"] = voc07_out

    # VOC2012 full
    voc12_n = len(list_images(VOC2012_JPEG))
    voc12_out = CACHE / f"voc2012_{DETECTOR}_n{voc12_n}_c{CONF}.json"
    if not voc12_out.exists():
        legacy = LEGACY_GEO_CACHE / f"voc2012_{DETECTOR}_n17125_c{CONF}.json"
        if legacy.exists() and voc12_n == 17125:
            remap_cache_by_basename(legacy, [VOC2012_JPEG], voc12_out)
        else:
            detect_folder_chunked(VOC2012_JPEG, voc12_out, voc_mode=True)
    out_paths["VOC2012"] = voc12_out

    # COCO train + val (separate then merge)
    n_train = len(list_images(COCO_TRAIN))
    n_val = len(list_images(COCO_VAL))
    coco_train_out = CACHE / f"coco_train_{DETECTOR}_n{n_train}_c{CONF}.json"
    coco_val_out = CACHE / f"coco_val_{DETECTOR}_n{n_val}_c{CONF}.json"
    coco_merged = CACHE / f"coco2017_trainval_{DETECTOR}_n{n_train + n_val}_c{CONF}.json"

    if not coco_train_out.exists():
        legacy = LEGACY_GEO_CACHE / "coco_train_yolov8m_n118287_c0.5.json"
        if legacy.exists() and n_train == 118287:
            remap_cache_by_basename(legacy, [COCO_TRAIN], coco_train_out)
        else:
            detect_folder_chunked(COCO_TRAIN, coco_train_out, voc_mode=False, batch=8)

    if not coco_val_out.exists():
        legacy = LEGACY_GEO_CACHE / "coco_val_yolov8m_n5000_c0.5.json"
        if legacy.exists() and n_val == 5000:
            remap_cache_by_basename(legacy, [COCO_VAL], coco_val_out)
        else:
            detect_folder_chunked(COCO_VAL, coco_val_out, voc_mode=False, batch=8)

    if not coco_merged.exists():
        print("merge COCO train+val detections...")
        a = json.loads(coco_train_out.read_text(encoding="utf-8"))
        b = json.loads(coco_val_out.read_text(encoding="utf-8"))
        merged = merge_pred_dicts([a, b])
        coco_merged.write_text(json.dumps(merged), encoding="utf-8")
        print(f"  merged images={len(merged)}")
    out_paths["COCO2017"] = coco_merged

    for k, p in out_paths.items():
        n = len(json.loads(p.read_text(encoding="utf-8")))
        print(f"[ok] {k}: {p.name}  images={n}")
    return out_paths


if __name__ == "__main__":
    prepare_all()
