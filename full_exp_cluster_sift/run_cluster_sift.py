# -*- coding: utf-8 -*-
"""Cluster-SIFT experiment: SIFT-mean k-means + within-cluster SIFT-J (no global recall)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from full_exp_cluster_sift.build_clusters import build_cluster_index, match_sift_in_cluster
from full_exp_cluster_sift.config import (
    AREA_THRESH,
    CAPACITY_TRIALS,
    CACHE,
    DET_CACHES,
    LOGS,
    MAX_OBJECTS,
    METHOD_NAME,
    MSG_LENGTHS,
    N_CLUSTERS,
    RESULTS,
    ROBUST_BYTES,
    ROBUST_MESSAGES,
    SCHEME_JSON,
    SEED,
    SEGMENT_BITS,
)
from full_exp_sift.retrieval_sift import SIFTCache
from stego_exp.geo_atoms import attach_atoms, load_schemes
from stego_exp.lazy_images import LazyImages
from stego_exp.paper_attacks import (
    ATTACK_DISPLAY,
    PAPER_ATTACK_NAMES,
    apply_paper_attack,
)
from stego_exp.pipeline import CoverlessSystem, segment_accuracy
from stego_exp.run_experiment import eval_capacity, object_stats


def _filter_proposed(preds: dict[str, list], conf: float = 0.5) -> dict[str, list]:
    out = {}
    for path, dets in preds.items():
        rows = [d for d in dets if float(d.get("score", 0)) >= conf]
        rows = sorted(rows, key=lambda d: -float(d.get("area_ratio", 0)))
        kept = [d for d in rows if float(d.get("area_ratio", 0)) >= AREA_THRESH]
        rows = kept if kept else rows[:1]
        rows = rows[:MAX_OBJECTS]
        if rows:
            out[path] = rows
    return out


def recover_cluster_sift(system, attacked_images, aux, pad_zeros, path_to_img, index, sift_cache):
    if len(attacked_images) != len(aux):
        return None, 0.0, 0.0
    bits = ""
    hits_top1 = 0
    hits_pool = 0
    for img, rec in zip(attacked_images, aux):
        cands = index.candidates_by_cluster(rec.cluster)
        pool_hit = rec.path in set(cands)
        found = match_sift_in_cluster(img, cands, path_to_img, sift_cache, index=index)
        if found == rec.path:
            hits_top1 += 1
        if pool_hit:
            hits_pool += 1
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
    if pad_zeros:
        bits = bits[: max(0, len(bits) - pad_zeros)]
    if len(bits) % 8:
        bits = bits[: len(bits) - (len(bits) % 8)]
    try:
        msg = int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""
    except Exception:
        msg = None
    return msg, hits_top1 / n, hits_pool / n


def eval_robustness(system, path_to_img, index, n_messages, msg_bytes, seed=49):
    rng = np.random.default_rng(seed)
    hides = []
    for _ in range(n_messages):
        msg = bytes(rng.integers(0, 256, size=msg_bytes, dtype=np.uint8).tolist())
        res = system.hide(msg)
        if res is not None:
            hides.append((msg, res))
    if not hides:
        return {"error": "no successful hides"}
    print(f"  hides={len(hides)} avg_stego={np.mean([h[1].n_stego for h in hides]):.1f}")

    sift_cache = SIFTCache(max_items=16384)
    out = {}
    for atk in tqdm(PAPER_ATTACK_NAMES, desc="TableIX-ClusterSIFT", mininterval=5.0):
        t0 = time.time()
        accs, r1s, rpools = [], [], []
        for msg, res in hides:
            attacked = [apply_paper_attack(path_to_img[p], atk) for p in res.stego_paths]
            pred, r1, rpool = recover_cluster_sift(
                system, attacked, res.aux, res.pad_zeros, path_to_img, index, sift_cache
            )
            accs.append(
                segment_accuracy(res.message_bits, pred, system.segment_bits, res.pad_zeros)
            )
            r1s.append(r1)
            rpools.append(rpool)
        out[atk] = {
            "ACC": float(np.mean(accs)),
            "R@1": float(np.mean(r1s)),
            "R@pool": float(np.mean(rpools)),
            "display": ATTACK_DISPLAY.get(atk, atk),
            "seconds": time.time() - t0,
        }
        print(
            f"  {atk}: ACC={out[atk]['ACC']:.3f} R@1={out[atk]['R@1']:.3f} "
            f"R@pool={out[atk]['R@pool']:.3f} ({out[atk]['seconds']:.0f}s)",
            flush=True,
        )
        (RESULTS / "_robust_ckpt.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    out["_avg"] = {
        "ACC": float(np.mean([out[a]["ACC"] for a in PAPER_ATTACK_NAMES])),
        "R@1": float(np.mean([out[a]["R@1"] for a in PAPER_ATTACK_NAMES])),
        "R@pool": float(np.mean([out[a]["R@pool"] for a in PAPER_ATTACK_NAMES])),
    }
    geo, nongeo = PAPER_ATTACK_NAMES[:13], PAPER_ATTACK_NAMES[13:]
    out["_avg_geometry"] = {
        "ACC": float(np.mean([out[a]["ACC"] for a in geo])),
        "R@1": float(np.mean([out[a]["R@1"] for a in geo])),
    }
    out["_avg_nongeometry"] = {
        "ACC": float(np.mean([out[a]["ACC"] for a in nongeo])),
        "R@1": float(np.mean([out[a]["R@1"] for a in nongeo])),
    }
    return out


def run_one_dataset(name: str):
    print(f"\n########## {name} Cluster-SIFT ##########")
    preds = json.loads(DET_CACHES[name].read_text(encoding="utf-8"))
    schemes = load_schemes(SCHEME_JSON, name)
    det = attach_atoms(_filter_proposed(preds), schemes, use_geo=True)
    paths = sorted(p for p in det if Path(p).exists())
    det = {p: det[p] for p in paths}
    print(f"usable={len(paths)} k={N_CLUSTERS[name]}")

    lazy = LazyImages(max_cache=128)
    lazy.add_paths(paths)
    index = build_cluster_index(name, paths, lazy)

    system = CoverlessSystem(
        detections=det,
        path_to_img=lazy,
        bank=index,
        segment_bits=SEGMENT_BITS,
        area_ratio_thresh=0.0,
        max_objects=MAX_OBJECTS,
        use_attributes=False,
        retrieve_topk=1,
    )
    t0 = time.time()
    stats = object_stats(system)
    print("stats", stats)
    cap = eval_capacity(system, MSG_LENGTHS, CAPACITY_TRIALS, seed=SEED)
    print("capacity", json.dumps(cap, indent=2))
    rob = eval_robustness(system, lazy, index, ROBUST_MESSAGES, ROBUST_BYTES, seed=SEED + 7)
    print("robust avg", rob.get("_avg"))
    return {
        "dataset": name,
        "method": METHOD_NAME,
        "protocol": "flat SIFT-mean k-means + within-cluster SIFT-J (no hierarchical CID, no global recall)",
        "n_images": len(paths),
        "n_clusters": N_CLUSTERS[name],
        "stats": stats,
        "capacity": cap,
        "robustness": rob,
        "seconds": time.time() - t0,
    }


def write_report(all_res: dict, path: Path):
    lines = [
        "# Cluster + SIFT-J Matching Results (参考论文1结构 × 参考论文3特征/匹配)",
        "",
        "- Clustering: k-means++ on normalized mean SIFT descriptors (flat, not hierarchical CID)",
        f"- K: VOC={N_CLUSTERS['VOC2007']}, COCO={N_CLUSTERS['COCO2017']} (paper-1 style scale)",
        "- Retrieval: **known cluster ID only** → Lowe+RANSAC maximize J inside cluster",
        "- No global coarse recall",
        "",
    ]
    for name, res in all_res.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"images={res['n_images']}, k={res['n_clusters']}, "
            f"atoms={res['stats']['codebook_size_atoms']}, objs/img={res['stats']['avg_objects']:.2f}"
        )
        lines.append("")
        lines.append("### Capacity (avg stego ↓)")
        lines.append("")
        lines.append("| 10B | 100B | 1KB | 2KB |")
        lines.append("|---:|---:|---:|---:|")

        def fmt(L):
            v = res["capacity"].get(str(L), {}).get("avg_stego")
            return f"{v:.1f}" if v is not None else "fail"

        lines.append(f"| {fmt(10)} | {fmt(100)} | {fmt(1000)} | {fmt(2000)} |")
        lines.append("")
        rob = res["robustness"]
        if "error" in rob:
            lines.append("### Robustness: error")
        else:
            a, g, ng = rob["_avg"], rob["_avg_geometry"], rob["_avg_nongeometry"]
            lines.append("### Robustness (Table IX)")
            lines.append("")
            lines.append("| ACC | R@1 | R@pool | Geo ACC | Geo R@1 | NonGeo ACC | NonGeo R@1 |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|")
            lines.append(
                f"| {a['ACC']:.3f} | {a['R@1']:.3f} | {a['R@pool']:.3f} | "
                f"{g['ACC']:.3f} | {g['R@1']:.3f} | {ng['ACC']:.3f} | {ng['R@1']:.3f} |"
            )
            lines.append("")
            lines.append("| Attack | ACC | R@1 | R@pool |")
            lines.append("|---|---:|---:|---:|")
            for atk in PAPER_ATTACK_NAMES:
                row = rob[atk]
                lines.append(
                    f"| {row['display']} | {row['ACC']:.3f} | {row['R@1']:.3f} | {row['R@pool']:.3f} |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all(datasets=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    datasets = datasets or list(DET_CACHES.keys())
    all_res = {}
    partial = RESULTS / "cluster_sift_results.partial.json"
    if partial.exists():
        all_res = json.loads(partial.read_text(encoding="utf-8"))
        print("resume", list(all_res))
    for name in datasets:
        if name in all_res:
            print("skip", name)
            continue
        res = run_one_dataset(name)
        all_res[name] = res
        partial.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
    out_json = RESULTS / "cluster_sift_results.json"
    out_md = RESULTS / "cluster_sift_results.md"
    out_json.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(all_res, out_md)
    print("Wrote", out_md)
    return all_res


if __name__ == "__main__":
    run_all()
