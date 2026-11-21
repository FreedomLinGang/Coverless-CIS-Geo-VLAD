"""Object detection with torchvision Faster R-CNN (COCO pretrained)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

from .config import COCO_CLASS_NAMES, CACHE_DIR, ExpConfig


def load_detector(device: str = "cuda"):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    return model, weights


@torch.inference_mode()
def detect_image(model, image_path: Path, conf_thresh: float, device: str) -> list[dict[str, Any]]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    tensor = F.to_tensor(img).to(device)
    out = model([tensor])[0]
    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    objs = []
    for box, score, lab in zip(boxes, scores, labels):
        if float(score) < conf_thresh:
            continue
        name = COCO_CLASS_NAMES[int(lab)] if int(lab) < len(COCO_CLASS_NAMES) else "N/A"
        if name in ("N/A", "__background__"):
            continue
        x1, y1, x2, y2 = map(float, box)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        objs.append(
            {
                "label": name,
                "score": float(score),
                "box": [x1, y1, x2, y2],
                "area": area,
                "area_ratio": area / float(max(1, w * h)),
                "cx": (x1 + x2) * 0.5,
                "cy": (y1 + y2) * 0.5,
                "aspect": (y2 - y1) / max(1e-6, x2 - x1),
            }
        )
    objs.sort(key=lambda o: (-o["area"], o["cy"], o["cx"], o["label"]))
    return objs


def list_images(folder: Path, max_images: int, seed: int) -> list[Path]:
    paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if max_images and len(paths) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), size=max_images, replace=False)
        paths = [paths[i] for i in sorted(idx.tolist())]
    return paths


def build_detection_cache(cfg: ExpConfig, image_paths: list[Path]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"det_{cfg.dataset}_n{len(image_paths)}_c{cfg.conf_thresh}.json"
    if cache_path.exists():
        return cache_path

    model, _ = load_detector(cfg.device)
    records = {}
    for p in tqdm(image_paths, desc="Detecting"):
        try:
            objs = detect_image(model, p, cfg.conf_thresh, cfg.device)
            records[str(p)] = objs
        except Exception as e:
            records[str(p)] = []
            print(f"[warn] detect fail {p}: {e}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    return cache_path


def load_detection_cache(cache_path: Path) -> dict[str, list[dict]]:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
