"""Chunked YOLOv8m detection for large folders (COCO train) with resume."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from geo_refine.config import CACHE, COCO_TRAIN
from geo_refine.detectors import build_detector


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def main():
    folder = COCO_TRAIN
    paths = list_images(folder)
    conf = 0.5
    cache = CACHE / f"coco_train_yolov8m_n{len(paths)}_c0.5.json"
    CACHE.mkdir(parents=True, exist_ok=True)

    preds: dict = {}
    if cache.exists():
        preds = json.loads(cache.read_text(encoding="utf-8"))
        print(f"resume from cache: {len(preds)}/{len(paths)}")

    todo = [p for p in paths if str(p) not in preds]
    print(f"to detect: {len(todo)}")
    if not todo:
        print("already complete")
        return

    det = build_detector("yolov8m", device="cuda")
    chunk = 64  # small to avoid OOM
    batch = 4
    t0 = time.time()
    start_done = len(preds)
    for i in tqdm(range(0, len(todo), chunk), desc="chunks"):
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
            preds[str(p)] = det._parse_result(r, voc_mode=False)
        del results
        torch.cuda.empty_cache()
        if (i // chunk) % 10 == 0:
            cache.write_text(json.dumps(preds), encoding="utf-8")
            done = len(preds)
            elapsed = time.time() - t0
            rate = (done - start_done) / max(1e-6, elapsed)
            print(f"  checkpoint {done}/{len(paths)}  {rate:.1f} img/s")

    cache.write_text(json.dumps(preds), encoding="utf-8")
    print(f"done {len(preds)} images in {time.time()-t0:.1f}s → {cache}")


if __name__ == "__main__":
    main()
