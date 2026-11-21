# -*- coding: utf-8 -*-
"""Parameter sweeps mirroring paper-1 Section IV.C: segment length n and max objects M.

Capacity-only + lightweight robustness (subset attacks) on VOC2007 with cluster-SIFT.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from full_exp_cluster_sift.build_clusters import build_cluster_index, match_sift_in_cluster
from full_exp_cluster_sift.config import (
    AREA_THRESH,
    CACHE,
    DET_CACHES,
    RESULTS,
    SCHEME_JSON,
    SEED,
)
from full_exp_sift.retrieval_sift import SIFTCache
from stego_exp.geo_atoms import attach_atoms, load_schemes
from stego_exp.lazy_images import LazyImages
from stego_exp.paper_attacks import PAPER_ATTACK_NAMES, apply_paper_attack, ATTACK_DISPLAY
from stego_exp.pipeline import CoverlessSystem, segment_accuracy
from stego_exp.run_experiment import eval_capacity, object_stats

OUT = RESULTS / "param_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# Representative attacks for param robustness (full 22 is too heavy for multi-param)
PARAM_ATTACKS = [
    "center_crop_5",
    "edge_crop_20",
    "rotate_30",
    "rotate_50",
    "translate_80_50",
    "jpeg_90",
    "gauss_noise",
    "hist_eq",
]


def _filter(preds, conf=0.5, area=AREA_THRESH, max_obj=8):
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


def recover(system, attacked, aux, pad, path_to_img, index, sift_cache):
    if len(attacked) != len(aux):
        return None, 0.0
    bits = ""
    hits = 0
    for img, rec in zip(attacked, aux):
        cands = index.candidates_by_cluster(rec.cluster)
        found = match_sift_in_cluster(img, cands, path_to_img, sift_cache, index=index)
        if found == rec.path:
            hits += 1
        atoms = system._atoms_for(found) if found else []
        rank_to_obj = {rank: oi for oi, rank in enumerate(rec.order) if rank > 0}
        for rank in sorted(rank_to_obj):
            oi = rank_to_obj[rank]
            if oi >= len(atoms):
                bits += "0" * system.segment_bits
                continue
            code = system.codebook.get(atoms[oi], "0" * system.segment_bits)
            bits += code
    n = max(1, len(aux))
    if pad:
        bits = bits[: max(0, len(bits) - pad)]
    if len(bits) % 8:
        bits = bits[: len(bits) - (len(bits) % 8)]
    try:
        msg = int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""
    except Exception:
        msg = None
    return msg, hits / n


def util_rate(system):
    """Fraction of selected objects actually used across successful hides of 100B."""
    rng = np.random.default_rng(7)
    used, total = 0, 0
    for _ in range(20):
        msg = bytes(rng.integers(0, 256, size=100, dtype=np.uint8).tolist())
        res = system.hide(msg)
        if res is None:
            continue
        for rec in res.aux:
            n_obj = len(rec.order)
            n_used = sum(1 for r in rec.order if r > 0)
            used += n_used
            total += n_obj
    return float(used / total) if total else 0.0


def make_system(name, segment_bits, max_objects, lazy, index):
    preds = json.loads(DET_CACHES[name].read_text(encoding="utf-8"))
    schemes = load_schemes(SCHEME_JSON, name)
    det = attach_atoms(_filter(preds, max_obj=max_objects), schemes, use_geo=True)
    paths = [p for p in lazy.paths if p in det] if hasattr(lazy, "paths") else sorted(det)
    # use intersection with index
    paths = [p for p in index.paths if p in det]
    det = {p: det[p] for p in paths}
    system = CoverlessSystem(
        detections=det,
        path_to_img=lazy,
        bank=index,
        segment_bits=segment_bits,
        area_ratio_thresh=0.0,
        max_objects=max_objects,
        use_attributes=False,
        retrieve_topk=1,
    )
    return system


def eval_rob_light(system, lazy, index, n_msg=5, msg_bytes=50, seed=49):
    rng = np.random.default_rng(seed)
    hides = []
    for _ in range(n_msg):
        msg = bytes(rng.integers(0, 256, size=msg_bytes, dtype=np.uint8).tolist())
        res = system.hide(msg)
        if res is not None:
            hides.append((msg, res))
    if not hides:
        return {"error": "no hides"}
    sift_cache = SIFTCache(max_items=8192)
    out = {}
    for atk in PARAM_ATTACKS:
        accs = []
        for msg, res in hides:
            attacked = [apply_paper_attack(lazy[p], atk) for p in res.stego_paths]
            pred, _ = recover(system, attacked, res.aux, res.pad_zeros, lazy, index, sift_cache)
            accs.append(segment_accuracy(res.message_bits, pred, system.segment_bits, res.pad_zeros))
        out[atk] = {"ACC": float(np.mean(accs)), "display": ATTACK_DISPLAY.get(atk, atk)}
    out["_avg"] = {"ACC": float(np.mean([out[a]["ACC"] for a in PARAM_ATTACKS]))}
    return out


def main():
    name = "VOC2007"
    print("==== param sweep", name, "====")
    preds = json.loads(DET_CACHES[name].read_text(encoding="utf-8"))
    schemes = load_schemes(SCHEME_JSON, name)
    det0 = attach_atoms(_filter(preds, max_obj=8), schemes, use_geo=True)
    paths = sorted(p for p in det0 if Path(p).exists())
    lazy = LazyImages(max_cache=64)
    lazy.add_paths(paths)
    index = build_cluster_index(name, paths, lazy)

    # --- A: segment bits n (capacity always; rob for n in {4,6,8}) ---
    seg_rows = {}
    for n in [4, 5, 6, 7, 8]:
        print(f"\n## segment_bits={n}", flush=True)
        t0 = time.time()
        system = make_system(name, n, 8, lazy, index)
        stats = object_stats(system)
        cap = eval_capacity(system, (10, 100, 1000), 10, seed=SEED)
        util = util_rate(system)
        if n in (4, 6, 8):
            rob = eval_rob_light(system, lazy, index, n_msg=4, msg_bytes=40)
        else:
            rob = {"_avg": {"ACC": None}, "skipped": True}
        seg_rows[str(n)] = {
            "stats": stats,
            "capacity": cap,
            "util_rate": util,
            "robustness": rob,
            "seconds": time.time() - t0,
        }
        print("cap", {k: v.get("avg_stego") for k, v in cap.items()}, "util", util,
              "rob", rob.get("_avg"), flush=True)

    # --- B: max objects M (capacity always; rob for M in {2,4,8}) ---
    m_rows = {}
    for M in [2, 4, 6, 8, 10]:
        print(f"\n## max_objects={M}", flush=True)
        t0 = time.time()
        system = make_system(name, 6, M, lazy, index)
        stats = object_stats(system)
        cap = eval_capacity(system, (10, 100, 1000), 10, seed=SEED)
        util = util_rate(system)
        if M in (2, 4, 8):
            rob = eval_rob_light(system, lazy, index, n_msg=4, msg_bytes=40)
        else:
            rob = {"_avg": {"ACC": None}, "skipped": True}
        m_rows[str(M)] = {
            "stats": stats,
            "capacity": cap,
            "util_rate": util,
            "robustness": rob,
            "seconds": time.time() - t0,
        }
        print("cap", {k: v.get("avg_stego") for k, v in cap.items()}, "util", util,
              "rob", rob.get("_avg"), flush=True)

    all_res = {"dataset": name, "segment_bits": seg_rows, "max_objects": m_rows}
    (OUT / "param_sweep_voc2007.json").write_text(
        json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["# Parameter Sweep (VOC2007, cluster-SIFT)", "", "## Segment length n (bits), M=8", ""]
    lines.append("| n | codes | 10B | 100B | 1KB | util | rob_avg |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for n, r in seg_rows.items():
        c = r["capacity"]

        def f(L, c=c):
            v = c.get(str(L), {}).get("avg_stego")
            return f"{v:.1f}" if v is not None else "fail"

        ra = r["robustness"].get("_avg", {}).get("ACC")
        ras = f"{ra:.3f}" if ra is not None else "—"
        lines.append(
            f"| {n} | {2**int(n)} | {f(10)} | {f(100)} | {f(1000)} | "
            f"{r['util_rate']:.3f} | {ras} |"
        )
    lines += ["", "## Max objects M, n=6", ""]
    lines.append("| M | avg_obj | 10B | 100B | 1KB | util | rob_avg |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for M, r in m_rows.items():
        c = r["capacity"]

        def f(L, c=c):
            v = c.get(str(L), {}).get("avg_stego")
            return f"{v:.1f}" if v is not None else "fail"

        ra = r["robustness"].get("_avg", {}).get("ACC")
        ras = f"{ra:.3f}" if ra is not None else "—"
        lines.append(
            f"| {M} | {r['stats']['avg_objects']:.2f} | {f(10)} | {f(100)} | {f(1000)} | "
            f"{r['util_rate']:.3f} | {ras} |"
        )
    (OUT / "param_sweep_voc2007.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
