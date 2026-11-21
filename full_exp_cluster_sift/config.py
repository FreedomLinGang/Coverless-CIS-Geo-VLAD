# -*- coding: utf-8 -*-
"""Paper-1-style flat clustering + within-cluster SIFT-J matching (参考论文3).

Uses normalized mean SIFT descriptor for flat k-means (NOT hierarchical CID).
Within-cluster match: Lowe + RANSAC, maximize J.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
FULL_EXP = ROOT.parent / "full_exp"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

FULL_CACHE = FULL_EXP / "cache"
FULL_RESULTS = FULL_EXP / "results"
SCHEME_JSON = FULL_RESULTS / "geo_subclass_full.json"

DET_CACHES = {
    "VOC2007": FULL_CACHE / "voc2007_yolov8m_n9963_c0.5.json",
    "VOC2012": FULL_CACHE / "voc2012_yolov8m_n17125_c0.5.json",
    "COCO2017": FULL_CACHE / "coco2017_trainval_yolov8m_n123287_c0.5.json",
}

N_CLUSTERS = {
    "VOC2007": 256,
    "VOC2012": 256,
    "COCO2017": 1024,
}

AREA_THRESH = 0.015
MAX_OBJECTS = 8
SEGMENT_BITS = 6
MSG_LENGTHS = (10, 100, 1000, 2000)
CAPACITY_TRIALS = 10
ROBUST_MESSAGES = 10
ROBUST_BYTES = 50
SEED = 42
METHOD_NAME = "cluster_sift"
SIFT_SIZE = 256
SIFT_T = 0.7
