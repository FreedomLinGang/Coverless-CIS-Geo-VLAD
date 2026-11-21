# -*- coding: utf-8 -*-
"""No cluster-ID retrieval: SIFT VLAD-64 shortlist + SIFT-J refine."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

from full_exp_cluster_sift.config import (  # noqa: E402
    AREA_THRESH,
    DET_CACHES,
    MAX_OBJECTS,
    SCHEME_JSON,
    SEGMENT_BITS,
)

VLAD_K = 64
SHORTLIST = 40
PCA_DIM = 256
PCA_MIN_IMAGES = 40000
ROBUST_MESSAGES = 10
ROBUST_BYTES = 50
SEED = 42
METHOD_NAME = "vlad_sift"
DATASETS = ("VOC2007", "VOC2012", "COCO2017")
