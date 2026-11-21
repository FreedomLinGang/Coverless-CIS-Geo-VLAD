# -*- coding: utf-8 -*-
"""Ablation + reference-reproduction baseline on VOC2007 (same gallery 9673).

Configs:
  C0  reference-repro: Faster R-CNN conf>=0.9 + coarse labels + Top-4-by-area
       + ResNet-101 deep-hash k-means++ clusters + within-cluster SSIM   (n=4,6)
  C1  ours n=4:  adaptive geo + area>=1.5% (fallback) + M<=8 + VLAD      (n=4)
  C2  coarse   :  coarse labels + Top-4 + VLAD                            (n=4)
  C3  uniform  :  uniform 4x8x4 geo + Top-4 + VLAD                        (n=6)
  C4  adaptive :  adaptive geo + Top-4 + VLAD                             (n=6)
  C5  ours n=6:  adaptive geo + area>=1.5% + M<=8 + VLAD                  (n=6)

Protocol: capacity 10/100/1000 B x 5 trials (SEED=42);
          robustness 8 representative attacks x 4 msgs x 40 B (seed=49).
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "geo_refine"))

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from tqdm import tqdm  # noqa: E402

from full_exp_cluster_sift.build_clusters import build_cluster_index  # noqa: E402
from full_exp_cluster_sift.config import DET_CACHES, SCHEME_JSON  # noqa: E402
from full_exp_cluster_sift.run_param_sweep import (  # noqa: E402
    _filter,
    eval_rob_light,
    PARAM_ATTACKS,
)
from full_exp_vlad_sift.build_vlad import build_vlad_retriever  # noqa: E402
from stego_exp.codebook import build_codebook, select_objects_proposed  # noqa: E402
from stego_exp.geo_atoms import attach_atoms, load_schemes  # noqa: E402
from stego_exp.lazy_images import LazyImages  # noqa: E402
from stego_exp.paper_attacks import ATTACK_DISPLAY, apply_paper_attack  # noqa: E402
from stego_exp.pipeline import CoverlessSystem, segment_accuracy  # noqa: E402
from stego_exp.retrieval import ssim_score  # noqa: E402
from stego_exp.run_experiment import eval_capacity, object_stats  # noqa: E402

OUT = Path(
    r"D:\a_python_codes\论文5\论文投稿打包_VLAD几何细分类_20260825"
    r"\02_实验结果\ablation\ablation_voc2007.json"
)
OUT.mkdir(parents=True, exist_ok=True)


class StubBank:
    """minimal bank for CoverlessSystem (cluster unused in VLAD/SSIM paths)."""
    def __init__(self, clusters: dict[str, int]):
        self._c = clusters

    def cluster_of(self, path: str) -> int:
        return self._c.get(path, 0)


def topk_dets(preds, conf, top=4):
    out = {}
    for path, dets in preds.items():
        rows = [d for d in dets if float(d.get("score", 0)) >= conf]
        rows = sorted(rows, key=lambda d: -float(d.get("area_ratio", 0)))[:top]
        if rows:
            out[path] = rows
    return out


def proposed_dets(preds, conf=0.5, area=0.015, max_obj=8):
    out = {}
    for path, dets in preds.items():
        rows = [d for d in dets if float(d.get("score", 0)) >= conf]
        rows = sorted(rows, key=lambda d: -float(d.get("area_ratio", 0)))
        kept = [d for d in rows if float(d.get("area_ratio", 0)) >= area]
        rows = kept if kept else rows[:1]
        rows = rows[:max_obj]
        if rows:
            out[path] = rows
    return out


def uniform_schemes(preds, min_count=35):
    """Per-class uniform 4x8x4 scheme dicts (mirrors reference_uniform)."""
    from geo_refine.adaptive_scheme import merge_sparse
    from geo_refine.geometry import GeoFeatures, subclass_id

    by = {}
    for path, dets in preds.items():
        for d in dets:
            by.setdefault(d["label"], []).append(d)
    schemes = {}
    for lab, items in by.items():
        c = Counter()
        for d in items:
            geo = GeoFeatures(
                d["diag_angle"], d["spatial_angle"], d["dist_ratio"],
                d["aspect"], d["area_ratio"],
            )
            c[subclass_id(lab, geo, 4, 8, [0.25, 0.5, 0.75], True, True, True)] += 1
        merged = merge_sparse(c, min_count)
        schemes[lab] = {
            "diag_n": 4, "spatial_n": 8, "dist_edges": [0.25, 0.5, 0.75],
            "use_diag": True, "use_spatial": True, "use_dist": True,
            "subclass_counts": dict(merged),
            "tier": "uniform",
        }
    return schemes


def make_system(det, path_to_img, segment_bits, max_objects, area_thresh, clusters=None):
    bank = StubBank(clusters or {})
    sys0 = CoverlessSystem(
        detections=det,
        path_to_img=path_to_img,
        bank=bank,
        segment_bits=segment_bits,
        area_ratio_thresh=area_thresh,
        max_objects=max_objects,
        use_attributes=False,
        retrieve_topk=40,
    )
    return sys0


def eval_rob_vlad(system, vlad, lazy, n_msg=4, msg_bytes=40, seed=49):
    rng = np.random.default_rng(seed)
    hides = []
    for _ in range(n_msg):
        msg = bytes(rng.integers(0, 256, size=msg_bytes, dtype=np.uint8).tolist())
        res = system.hide(msg)
        if res is not None:
            hides.append((msg, res))
    if not hides:
        return {"error": "no hides"}
    out = {}
    for atk in PARAM_ATTACKS:
        accs = []
        for msg, res in hides:
            attacked = [apply_paper_attack(lazy[p], atk) for p in res.stego_paths]
            pred, r1, rp = vlad.recover_with_recall(system, attacked, res.aux, res.pad_zeros, lazy)
            accs.append(segment_accuracy(res.message_bits, pred, system.segment_bits, res.pad_zeros))
        out[atk] = {"ACC": float(np.mean(accs)), "display": ATTACK_DISPLAY.get(atk, atk)}
    out["_avg"] = {"ACC": float(np.mean([out[a]["ACC"] for a in PARAM_ATTACKS]))}
    return out


def eval_rob_ssim_cluster(system, clusters_f, lazy, n_msg=4, msg_bytes=40, seed=49):
    """Reference-style: within-cluster SSIM (all cluster members)."""
    from collections import defaultdict

    members = defaultdict(list)
    for p, c in clusters_f.items():
        members[c].append(p)
    rng = np.random.default_rng(seed)
    hides = []
    for _ in range(n_msg):
        msg = bytes(rng.integers(0, 256, size=msg_bytes, dtype=np.uint8).tolist())
        res = system.hide(msg)
        if res is not None:
            hides.append((msg, res))
    if not hides:
        return {"error": "no hides"}
    out = {}
    for atk in PARAM_ATTACKS:
        accs, r1s = [], []
        for msg, res in hides:
            bits = ""
            hits = 0
            for p, rec in zip(res.stego_paths, res.aux):
                q = apply_paper_attack(lazy[p], atk)
                cands = members.get(rec.cluster, [])
                best_p, best_s = None, -1.0
                for cp in cands:
                    s = ssim_score(q, lazy[cp])
                    if s > best_s:
                        best_s, best_p = s, cp
                if best_p == rec.path:
                    hits += 1
                atoms = system._atoms_for(best_p) if best_p else []
                rank_to_obj = {rank: oi for oi, rank in enumerate(rec.order) if rank > 0}
                for rank in sorted(rank_to_obj):
                    oi = rank_to_obj[rank]
                    if oi >= len(atoms):
                        bits += "0" * system.segment_bits
                        continue
                    bits += system.codebook.get(atoms[oi], "0" * system.segment_bits)
            n = max(1, len(res.aux))
            if res.pad_zeros:
                bits = bits[: max(0, len(bits) - res.pad_zeros)]
            if len(bits) % 8:
                bits = bits[: len(bits) - (len(bits) % 8)]
            try:
                pred = int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""
            except Exception:
                pred = None
            accs.append(segment_accuracy(res.message_bits, pred, system.segment_bits, res.pad_zeros))
            r1s.append(hits / n)
        out[atk] = {"ACC": float(np.mean(accs)), "display": ATTACK_DISPLAY.get(atk, atk)}
    out["_avg"] = {"ACC": float(np.mean([out[a]["ACC"] for a in PARAM_ATTACKS]))}
    return out


def resnet_hash_clusters(paths, lazy, k=256, device="cuda", batch=96):
    """ResNet-101(ImageNet) features -> min-max normalize -> binary hash (Eq.1 of ref)
    -> k-means++ on sequences -> {path: cluster}."""
    import torchvision

    torch.cuda.set_device(0)
    model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
    model = model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch), desc="ResNet-101"):
            imgs = []
            for p in paths[i : i + batch]:
                im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
                imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
            x = torch.from_numpy(np.stack(imgs, 0)).permute(0, 3, 1, 2).to(device)
            x = (x - mean) / std
            f = model(x).cpu().numpy()  # 2048-d
            feats.append(f)
    feats = np.concatenate(feats, 0).astype(np.float32)
    # min-max normalize per dim (reference Eq.1)
    mn, mx = feats.min(0), feats.max(0)
    norm = (feats - mn) / np.maximum(mx - mn, 1e-8)
    h = (norm > 0.5).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=k, init="k-means++", random_state=42,
                         batch_size=2048, n_init=3, max_iter=200)
    labels = km.fit_predict(h)
    return {p: int(labels[i]) for i, p in enumerate(paths)}, km, norm, h


def main():
    t_all = time.time()
    name = "VOC2007"
    preds_yolo = json.loads(DET_CACHES[name].read_text(encoding="utf-8"))
    schemes_ada = load_schemes(SCHEME_JSON, name)
    det_prop = attach_atoms(proposed_dets(preds_yolo), schemes_ada, use_geo=True)
    paths = sorted(p for p in det_prop if Path(p).exists())
    print("gallery n =", len(paths), flush=True)
    lazy = LazyImages(max_cache=96)
    lazy.add_paths(paths)

    # build VLAD retriever (cached bank)
    vlad = build_vlad_retriever(name, paths, lazy, shortlist=40)

    # faster R-CNN cache (geo_refine/cache)
    fr = json.loads(
        Path(r"D:\a_python_codes\论文5\our_code\geo_refine\cache"
             r"\voc2007_fasterrcnn_n9963_c0.5.json").read_text(encoding="utf-8")
    )
    fr_dets = {p: ds for p, ds in fr.items() if p in set(paths)}
    fr_top4 = attach_atoms(topk_dets(fr_dets, 0.9, top=4), None, use_geo=False)
    fr_paths = sorted(p for p in fr_top4 if Path(p).exists())
    print("FRCNN kept images:", len(fr_paths), flush=True)

    # ---- C0: reference-repro (ResNet-101 hash clusters + within-cluster SSIM) ----
    clusters_f, km_c, norm_f, hash_f = resnet_hash_clusters(paths, lazy, k=256)
    print("deep-hash clusters built", flush=True)
    c0 = {}
    for n in (4, 6):
        sys0 = make_system(fr_top4, lazy, segment_bits=n, max_objects=4,
                           area_thresh=0.0, clusters=clusters_f)
        stats = object_stats(sys0)
        cap = eval_capacity(sys0, (10, 100, 1000), 5, seed=42)
        print(f"C0 n={n} cap", {k: v.get("avg_stego") for k, v in cap.items()},
              "succ", {k: v.get("success_rate") for k, v in cap.items()}, flush=True)
        rob = None
        if n == 4:
            rob = eval_rob_ssim_cluster(sys0, clusters_f, lazy, n_msg=4, msg_bytes=40)
            print("C0 n=4 rob_avg", rob.get("_avg"), flush=True)
        c0[str(n)] = {"stats": stats, "capacity": cap, "robustness": rob}
    json.dump({"c0": c0}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- uniform geometric schemes (4x8x4, merged at tau=35) ----
    schemes_uni = uniform_schemes(
        {p: [d for d in ds] for p, ds in proposed_dets(preds_yolo).items()}, 35
    )
    det_uni = attach_atoms(topk_dets(preds_yolo, 0.5, top=4), schemes_uni, use_geo=True)

    # ---- configs: (label, det, segment_bits, max_objects, area_thresh) ----
    yolo_top4 = attach_atoms(topk_dets(preds_yolo, 0.5, top=4), None, use_geo=False)
    ada_top4 = attach_atoms(topk_dets(preds_yolo, 0.5, top=4), schemes_ada, use_geo=True)
    dets_prop8 = attach_atoms(proposed_dets(preds_yolo), schemes_ada, use_geo=True)

    configs = [
        ("C1_ours4", dets_prop8, 4, 8, 0.015),
        ("C2_coarse4", yolo_top4, 4, 4, 0.0),
        ("C3_uniform6", det_uni, 6, 4, 0.0),
        ("C4_ada_top4_6", ada_top4, 6, 4, 0.0),
        ("C5_ours6", dets_prop8, 6, 8, 0.015),
    ]
    res = {}
    for label, det, n, M, ath in configs:
        t0 = time.time()
        print(f"\n==== {label} n={n} M={M} ====", flush=True)
        sysx = make_system(det, lazy, segment_bits=n, max_objects=M, area_thresh=ath)
        stats = object_stats(sysx)
        cap = eval_capacity(sysx, (10, 100, 1000), 5, seed=42)
        rob = eval_rob_vlad(sysx, vlad, lazy, n_msg=4, msg_bytes=40)
        res[label] = {
            "segment_bits": n, "max_objects": M, "stats": stats, "capacity": cap,
            "robustness": rob, "seconds": time.time() - t0,
        }
        print(label, "cap", {k: (v.get("avg_stego"), v.get("success_rate")) for k, v in cap.items()},
              "rob", rob.get("_avg"), flush=True)
        json.dump({"c0": c0, "configs": res}, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    json.dump({"c0": c0, "configs": res}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("wrote", OUT, "total", round(time.time() - t_all, 1), "s", flush=True)


if __name__ == "__main__":
    main()
