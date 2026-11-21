# -*- coding: utf-8 -*-
"""Coverless recover without cluster ID: VLAD shortlist + SIFT-J."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from full_exp_cluster_sift.build_clusters import build_cluster_index
from full_exp_cluster_sift.config import MAX_OBJECTS, SCHEME_JSON, SEGMENT_BITS
from full_exp_cluster_sift.run_cluster_sift import _filter_proposed
from full_exp_vlad_sift.build_vlad import build_vlad_retriever
from full_exp_vlad_sift.config import (
    CACHE,
    DATASETS,
    DET_CACHES,
    LOGS,
    METHOD_NAME,
    RESULTS,
    ROBUST_BYTES,
    ROBUST_MESSAGES,
    SEED,
    SHORTLIST,
    VLAD_K,
)
from no_aux_hash.gallery import load_gallery
from stego_exp.geo_atoms import attach_atoms, load_schemes
from stego_exp.paper_attacks import ATTACK_DISPLAY, PAPER_ATTACK_NAMES, apply_paper_attack
from stego_exp.pipeline import CoverlessSystem, recover_with_recall, segment_accuracy
from stego_exp.run_experiment import eval_capacity, object_stats
from full_exp_cluster_sift.config import MSG_LENGTHS, CAPACITY_TRIALS


def eval_robustness(system, path_to_img, n_messages, msg_bytes, seed=49):
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
    out = {}
    ckpt = RESULTS / "_robust_ckpt.json"
    if ckpt.exists():
        prev = json.loads(ckpt.read_text(encoding="utf-8"))
        if prev.get("_dataset") == getattr(eval_robustness, "_name", ""):
            out = {k: v for k, v in prev.items() if not str(k).startswith("_")}
            print("  resume attacks", list(out))
    for atk in tqdm(PAPER_ATTACK_NAMES, desc="TableIX-VLAD", mininterval=5.0):
        if atk in out:
            continue
        t0 = time.time()
        accs, r1s, rpools = [], [], []
        for msg, res in hides:
            attacked = [apply_paper_attack(path_to_img[p], atk) for p in res.stego_paths]
            pred, r1, rpool = recover_with_recall(
                system, attacked, res.aux, res.pad_zeros, path_to_img
            )
            accs.append(segment_accuracy(res.message_bits, pred, system.segment_bits, res.pad_zeros))
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
        payload = dict(out)
        payload["_dataset"] = getattr(eval_robustness, "_name", "")
        ckpt.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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


def run_one_dataset(name: str, skip_capacity: bool = True):
    print(f"\n########## {name} VLAD-SIFT (no cluster ID) ##########")
    paths, images, index, centers_km, labels = load_gallery(name)
    vlad = build_vlad_retriever(name, paths, images, shortlist=SHORTLIST, k=VLAD_K)
    preds = json.loads(DET_CACHES[name].read_text(encoding="utf-8"))
    schemes = load_schemes(SCHEME_JSON, name)
    det = attach_atoms(_filter_proposed(preds), schemes, use_geo=True)
    det = {p: det[p] for p in paths}
    g = vlad.gallery
    print(
        f"usable={len(paths)} vlad_k={VLAD_K} shortlist={SHORTLIST} "
        f"search_dim={g.shape[1]}"
    )

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
    t0 = time.time()
    stats = object_stats(system)
    print("stats", stats)
    cap = None
    if not skip_capacity:
        cap = eval_capacity(system, MSG_LENGTHS, CAPACITY_TRIALS, seed=SEED)
        print("capacity", json.dumps(cap, indent=2))
    eval_robustness._name = name
    rob = eval_robustness(system, images, ROBUST_MESSAGES, ROBUST_BYTES, seed=SEED + 7)
    print("robust avg", rob.get("_avg"))
    return {
        "dataset": name,
        "method": METHOD_NAME,
        "protocol": "SIFT VLAD-64 cosine shortlist + SIFT-J; no transmitted cluster/class ID",
        "n_images": len(paths),
        "vlad_k": VLAD_K,
        "shortlist": SHORTLIST,
        "search_dim": int(g.shape[1]),
        "stats": stats,
        "capacity": cap,
        "robustness": rob,
        "seconds": time.time() - t0,
    }


def write_report(all_res: dict, path: Path):
    lines = [
        "# VLAD + SIFT-J（无簇/类别辅助信息）",
        "",
        "- 粗检索：SIFT 描述子 VLAD-64，全库余弦 top-40",
        "- 精排：Lowe+RANSAC 最大化 J（与 Cluster-SIFT 相同）",
        "- **不使用**辅助信息中的簇ID或类别",
        "",
    ]
    for name, res in all_res.items():
        lines.append(f"## {name}")
        lines.append("")
        dim = res.get("search_dim", 8192)
        lines.append(
            f"images={res['n_images']}, vlad_k={res['vlad_k']}, "
            f"shortlist={res['shortlist']}, search_dim={dim}"
        )
        lines.append("")
        rob = res["robustness"]
        if "error" in rob:
            lines.append("robustness error")
        else:
            a, g, ng = rob["_avg"], rob["_avg_geometry"], rob["_avg_nongeometry"]
            lines.append("| ACC | R@1 | R@pool | Geo ACC | NonGeo ACC |")
            lines.append("|---:|---:|---:|---:|---:|")
            lines.append(
                f"| {a['ACC']:.3f} | {a['R@1']:.3f} | {a['R@pool']:.3f} | {g['ACC']:.3f} | {ng['ACC']:.3f} |"
            )
            lines.append("")
            lines.append("| Attack | ACC | R@1 | R@pool |")
            lines.append("|---|---:|---:|---:|")
            for atk in PAPER_ATTACK_NAMES:
                row = rob[atk]
                lines.append(f"| {row['display']} | {row['ACC']:.3f} | {row['R@1']:.3f} | {row['R@pool']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all(datasets=None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    datasets = list(datasets or DATASETS)
    partial = RESULTS / "vlad_sift_results.partial.json"
    all_res = {}
    if partial.exists():
        all_res = json.loads(partial.read_text(encoding="utf-8"))
        print("resume", list(all_res))
    for name in datasets:
        if name in all_res and "error" not in all_res[name].get("robustness", {}):
            print("skip", name)
            continue
        res = run_one_dataset(name)
        all_res[name] = res
        partial.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
        ckpt = RESULTS / "_robust_ckpt.json"
        if ckpt.exists():
            ckpt.unlink()
    out_json = RESULTS / "vlad_sift_results.json"
    out_md = RESULTS / "vlad_sift_results.md"
    out_json.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(all_res, out_md)
    print("Wrote", out_md)
    return all_res


if __name__ == "__main__":
    import sys

    ds = [a for a in sys.argv[1:] if not a.startswith("-")]
    run_all(ds or None)
