# -*- coding: utf-8 -*-
"""Example 2/4/5 support: codebook codes for the example image 007332.jpg and
actual no-cluster VLAD-SIFT retrieval verification under attacks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\a_python_codes\论文5\our_code")
sys.path.insert(0, str(ROOT))

from full_exp_cluster_sift.config import AREA_THRESH, DET_CACHES, SCHEME_JSON  # noqa: E402
from full_exp_vlad_sift.build_vlad import build_vlad_retriever  # noqa: E402
from no_aux_hash.gallery import load_gallery  # noqa: E402
from stego_exp.codebook import build_codebook  # noqa: E402
from stego_exp.geo_atoms import attach_atoms, load_schemes  # noqa: E402
from stego_exp.paper_attacks import apply_paper_attack  # noqa: E402


def main():
    name = "VOC2007"
    preds = json.loads(DET_CACHES[name].read_text(encoding="utf-8"))
    schemes = load_schemes(SCHEME_JSON, name)

    def filt(preds):
        out = {}
        for p, raw in preds.items():
            rows = [d for d in raw if float(d.get("score", 0)) >= 0.5]
            rows = sorted(rows, key=lambda d: -float(d.get("area_ratio", 0)))
            kept = [d for d in rows if float(d.get("area_ratio", 0)) >= AREA_THRESH]
            rows = kept if kept else rows[:1]
            rows = rows[:8]
            if rows:
                out[p] = rows
        return out

    det = attach_atoms(filt(preds), schemes, use_geo=True)
    codebook = build_codebook(det, 6, use_attributes=False, image_loader=None)

    ex = r"D:\datasets\VOC2007\VOC2007_test\JPEGImages\007332.jpg"
    objs = det[ex]
    print("== example image:", Path(ex).name)
    for i, o in enumerate(objs):
        code = codebook.get(o["atom"], "??????")
        print(f"  #{i+1} {o['atom']:24s} area={o['area_ratio']:.4f} conf={o['score']:.3f} -> {code}")
    msg = "".join(codebook[o["atom"]] for o in objs[:6])
    print("first 6 codes concat:", msg, "-> bytes:", len(msg) // 8)
    n_obj = len(objs)
    print("n_obj:", n_obj, "obj coverage (images/code):",
          {codebook[o["atom"]]: None for o in objs})

    # ---- retrieval verification (no cluster ID) ----
    paths, images, index, _c, _l = load_gallery(name)
    vlad = build_vlad_retriever(name, paths, images, shortlist=40)
    from PIL import Image

    img = Image.open(ex).convert("RGB")
    for atk in ("none", "rotate_50", "edge_crop_20", "hist_eq"):
        q = img if atk == "none" else apply_paper_attack(img, atk)
        cands = vlad.shortlist_paths(q)
        rank_of = (cands.index(ex) + 1) if ex in cands else None
        found = vlad.match_path(q, images, cands)
        print(f"  [{atk}] rank_in_top40={rank_of} matched={Path(found).name} ok={found == ex}")


if __name__ == "__main__":
    main()
