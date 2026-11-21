"""Paths and experiment defaults for geometric fine-grained detection study."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

# Pascal VOC
VOC2007_TRAIN = Path(r"D:\datasets\VOC2007\VOC2007_train")
VOC2007_TEST = Path(r"D:\datasets\VOC2007\VOC2007_test")
VOC2012_ROOT = Path(r"D:\datasets\VOC2012")

# Legacy aliases (VOC2012)
VOC_ROOT = VOC2012_ROOT
VOC_JPEG = VOC_ROOT / "JPEGImages"
VOC_ANN = VOC_ROOT / "Annotations"

# MS COCO 2017 (images nested under train2017/train2017 etc.)
COCO_ROOT = Path(r"D:\datasets\coco")
COCO_TRAIN = COCO_ROOT / "train2017" / "train2017"
COCO_VAL = COCO_ROOT / "val2017" / "val2017"
COCO_ANN_TRAIN = COCO_ROOT / "annotations_trainval2017" / "annotations" / "instances_train2017.json"
COCO_ANN_VAL = COCO_ROOT / "annotations_trainval2017" / "annotations" / "instances_val2017.json"

# Deprecated incomplete RGB subset (kept only if other scripts import)
COCO_RGB = Path(r"D:\datasets\coco_rgb")

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

DIAG_BINS_REF = 4
SPATIAL_BINS_REF = 8
DIST_BINS_REF = 4
MIN_COUNT_PER_SUBCLASS = 30
FREQ_CLASS_RATIO = 0.03
MED_CLASS_RATIO = 0.01


@dataclass
class RunConfig:
    device: str = "cuda"
    conf_thresh: float = 0.5
    iou_eval: float = 0.5
    max_images_voc: int = 800
    max_images_coco: int = 800
    seed: int = 42
    detectors: tuple[str, ...] = ("fasterrcnn", "yolov8s", "yolov8m")
    skip_coco_map: bool = True
