"""Detector wrappers: Faster R-CNN and YOLOv8."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as F

from .config import VOC_CLASSES
from .geometry import detection_record

# torchvision COCO name list (same as earlier project)
COCO_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# map COCO-ish names → VOC names for VOC evaluation
COCO_TO_VOC = {
    "airplane": "aeroplane",
    "motorcycle": "motorbike",
    "potted plant": "pottedplant",
    "couch": "sofa",
    "tv": "tvmonitor",
    "dining table": "diningtable",
}


def normalize_label(name: str, voc_mode: bool) -> str | None:
    name = name.strip()
    if name in ("N/A", "__background__", ""):
        return None
    if not voc_mode:
        return name
    mapped = COCO_TO_VOC.get(name, name)
    if mapped in VOC_CLASSES:
        return mapped
    return None


class BaseDetector:
    name: str = "base"

    def detect(self, image_path: Path, conf: float, voc_mode: bool) -> list[dict[str, Any]]:
        raise NotImplementedError


class FasterRCNNDetector(BaseDetector):
    name = "fasterrcnn"

    def __init__(self, device: str = "cuda"):
        import torchvision

        self.device = device
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(device).eval()

    @torch.inference_mode()
    def detect(self, image_path: Path, conf: float, voc_mode: bool) -> list[dict[str, Any]]:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        t = F.to_tensor(img).to(self.device)
        out = self.model([t])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        recs = []
        for box, score, lab in zip(boxes, scores, labels):
            if float(score) < conf:
                continue
            raw = COCO_NAMES[int(lab)] if int(lab) < len(COCO_NAMES) else "N/A"
            name = normalize_label(raw, voc_mode)
            if name is None:
                continue
            recs.append(detection_record(name, float(score), list(map(float, box)), w, h))
        return recs


class YOLODetector(BaseDetector):
    def __init__(self, weights: str = "yolov8s.pt", device: str = "cuda"):
        from ultralytics import YOLO

        self.name = Path(weights).stem
        self.device = device
        self.model = YOLO(weights)

    def _parse_result(self, r0, voc_mode: bool) -> list[dict[str, Any]]:
        recs = []
        if r0 is None or r0.boxes is None:
            return recs
        # original image size
        h, w = int(r0.orig_shape[0]), int(r0.orig_shape[1])
        names = r0.names
        for b in r0.boxes:
            score = float(b.conf.item())
            cls_id = int(b.cls.item())
            raw = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            name = normalize_label(str(raw), voc_mode)
            if name is None:
                continue
            xyxy = b.xyxy.cpu().numpy().reshape(-1)
            recs.append(detection_record(name, score, [float(x) for x in xyxy], w, h))
        return recs

    @torch.inference_mode()
    def detect(self, image_path: Path, conf: float, voc_mode: bool) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return self._parse_result(results[0], voc_mode)

    @torch.inference_mode()
    def detect_folder(
        self,
        folder: Path,
        conf: float,
        voc_mode: bool,
        batch: int = 16,
    ) -> dict[str, list[dict[str, Any]]]:
        """Batched YOLO predict over a folder; keys are absolute path strings."""
        results = self.model.predict(
            source=str(folder),
            conf=conf,
            device=self.device,
            verbose=False,
            stream=True,
            batch=batch,
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            path = str(Path(r.path).resolve()) if getattr(r, "path", None) else None
            if path is None:
                continue
            out[path] = self._parse_result(r, voc_mode)
        return out


def build_detector(kind: str, device: str = "cuda") -> BaseDetector:
    kind = kind.lower()
    if kind in ("fasterrcnn", "frcnn"):
        return FasterRCNNDetector(device=device)
    if kind.startswith("yolo"):
        # yolov8n / yolov8s / yolov8m
        weight = kind if kind.endswith(".pt") else f"{kind}.pt"
        return YOLODetector(weights=weight, device=device)
    raise ValueError(f"Unknown detector: {kind}")
