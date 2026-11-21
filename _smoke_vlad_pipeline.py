# -*- coding: utf-8 -*-
"""Smoke test: CoverlessSystem(vlad_retriever=...) vs stego_VOC2007.json on one attack."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from full_exp_cluster_sift.config import DET_CACHES, MAX_OBJECTS, SCHEME_JSON, SEGMENT_BITS
from full_exp_cluster_sift.run_cluster_sift import _filter_proposed
from full_exp_vlad_sift.build_vlad import build_vlad_retriever
from no_aux_hash.gallery import load_gallery
from stego_exp.geo_atoms import attach_atoms, load_schemes
from stego_exp.paper_attacks import apply_paper_attack
from stego_exp.pipeline import CoverlessSystem, recover_with_recall, segment_accuracy

SEED = 49
MSG_BYTES = 50
ATTACK = "rotate_50"


def main():
    name = "VOC2007"
    ref_path = ROOT / "full_exp_vlad_sift" / "results" / "vlad_sift_results.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_acc = ref["VOC2007"]["robustness"]["rotate_50"]["ACC"]
    ref_r1 = ref["VOC2007"]["robustness"]["rotate_50"]["R@1"]

    paths, images, index, _, _ = load_gallery(name)
    preds = json.loads(DET_CACHES[name].read_text(encoding="utf-8"))
    schemes = load_schemes(SCHEME_JSON, name)
    det = attach_atoms(_filter_proposed(preds), schemes, use_geo=True)
    det = {p: det[p] for p in paths}

    vlad = build_vlad_retriever(name, paths, images, shortlist=40)
    system = CoverlessSystem(
        detections=det,
        path_to_img=images,
        bank=index,
        segment_bits=SEGMENT_BITS,
        area_ratio_thresh=0.0,
        max_objects=MAX_OBJECTS,
        use_attributes=False,
        retrieve_topk=1,
        vlad_retriever=vlad,
    )

    rng = np.random.default_rng(SEED)
    msg = bytes(rng.integers(0, 256, size=MSG_BYTES, dtype=np.uint8).tolist())
    res = system.hide(msg)
    assert res is not None, "hide failed"

    attacked = [apply_paper_attack(images[p], ATTACK) for p in res.stego_paths]
    pred, r1, rpool = recover_with_recall(system, attacked, res.aux, res.pad_zeros, images)
    acc = segment_accuracy(res.message_bits, pred, system.segment_bits, res.pad_zeros)

    print(f"attack={ATTACK} ACC={acc:.3f} R@1={r1:.3f} R@pool={rpool:.3f}")
    print(f"ref   ACC={ref_acc:.3f} R@1={ref_r1:.3f}")
    ok = abs(acc - ref_acc) < 0.05 and abs(r1 - ref_r1) < 0.05
    print("PASS" if ok else "CHECK (single-trial variance expected)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
