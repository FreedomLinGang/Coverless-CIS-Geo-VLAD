# -*- coding: utf-8 -*-
"""Micro-benchmark VLAD-SIFT recovery latency on VOC2012 (mirrors bench_timing.py)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from full_exp_vlad_sift.build_vlad import (  # noqa: E402
    build_vlad_retriever,
    gallery_scores,
)
from no_aux_hash.gallery import load_gallery  # noqa: E402
from no_aux_hash.he_index import vlad_vector  # noqa: E402
from no_aux_hash.hashes import topk_indices  # noqa: E402
from no_aux_hash.rich_features import detect_sift  # noqa: E402
from stego_exp.paper_attacks import apply_paper_attack  # noqa: E402

OUT = Path(
    r"D:\a_python_codes\论文5\论文投稿打包_VLAD几何细分类_20260825"
    r"\02_实验结果\timing\timing_voc2012.json"
)
N_QUERY = 16


def main():
    paths, images, index, _c, _l = load_gallery("VOC2012")
    vlad = build_vlad_retriever("VOC2012", paths, images, shortlist=40)
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(paths), size=N_QUERY, replace=False)

    # warmup
    img0 = images[paths[int(idxs[0])]]
    _ = vlad.shortlist_paths(img0)
    _ = vlad.match_path(img0, images)

    rows = {}
    for atk in ("none", "rotate_50", "edge_crop_20", "hist_eq"):
        enc, search, refine, total = [], [], [], []
        for i in idxs:
            img = images[paths[int(i)]]
            if atk != "none":
                img = apply_paper_attack(img, atk)

            t0 = time.perf_counter()
            _, desc = detect_sift(img)
            q = vlad_vector(desc, vlad.centers)
            t1 = time.perf_counter()
            scores = gallery_scores(vlad.gallery, q)
            idx = topk_indices(scores, vlad.shortlist, True)
            cands = [vlad.paths[j] for j in idx]
            t2 = time.perf_counter()
            _ = vlad.match_path(img, images, cands)
            t3 = time.perf_counter()

            enc.append((t1 - t0) * 1000)
            search.append((t2 - t1) * 1000)
            refine.append((t3 - t2) * 1000)
            total.append((t3 - t0) * 1000)

        rows[atk] = {
            "encode_ms": float(np.mean(enc)),
            "search_ms": float(np.mean(search)),
            "refine_ms": float(np.mean(refine)),
            "total_ms": float(np.mean(total)),
            "n": N_QUERY,
            "gallery": len(paths),
            "shortlist": 40,
            "vlad_dim": int(vlad.gallery.shape[1]),
        }
        print(atk, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[atk].items()},
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
