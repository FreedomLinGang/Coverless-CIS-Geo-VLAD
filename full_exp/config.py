# -*- coding: utf-8 -*-
"""Full-dataset proposed-method experiment.

Datasets (train+val/test combined):
  - D:\\datasets\\VOC2007  (train + test)
  - D:\\datasets\\VOC2012  (full JPEGImages)
  - D:\\datasets\\coco     (coco_train2017 + coco_val2017)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

# ---- datasets ----
VOC2007_TRAIN_JPEG = Path(r"D:\datasets\VOC2007\VOC2007_train\JPEGImages")
VOC2007_TEST_JPEG = Path(r"D:\datasets\VOC2007\VOC2007_test\JPEGImages")
VOC2012_JPEG = Path(r"D:\datasets\VOC2012\JPEGImages")
COCO_TRAIN = Path(r"D:\datasets\coco\coco_train2017")
COCO_VAL = Path(r"D:\datasets\coco\coco_val2017")

# Legacy detection caches (may use old COCO nested paths)
LEGACY_GEO_CACHE = Path(__file__).resolve().parents[1] / "geo_refine" / "cache"

# ---- proposed method defaults ----
DETECTOR = "yolov8m"
CONF = 0.5
AREA_THRESH = 0.015
MAX_OBJECTS = 8
SEGMENT_BITS = 6
N_CLUSTERS = {
    "VOC2007": 128,
    "VOC2012": 128,
    "COCO2017": 256,
}
MSG_LENGTHS = (10, 100, 1000, 2000)
CAPACITY_TRIALS = 10
ROBUST_MESSAGES = 10
ROBUST_BYTES = 50
RETRIEVE_TOPK = 40
DEVICE = "cuda"
SEED = 42
