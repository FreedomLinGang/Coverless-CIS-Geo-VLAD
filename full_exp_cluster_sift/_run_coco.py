# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from full_exp_cluster_sift.run_cluster_sift import run_all

if __name__ == "__main__":
    # VOC already done in partial; only run COCO2017
    run_all(["COCO2017"])
