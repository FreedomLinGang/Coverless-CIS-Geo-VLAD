# -*- coding: utf-8 -*-
"""Full-dataset capacity + Table-IX robustness for the proposed method."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from full_exp.build_banks import bank_path_for, build_bank_chunked
from full_exp.build_schemes import build_all_schemes
from full_exp.config import (
    AREA_THRESH,
    CAPACITY_TRIALS,
    CACHE,
    LOGS,
    MAX_OBJECTS,
    MSG_LENGTHS,
    N_CLUSTERS,
    RESULTS,
    RETRIEVE_TOPK,
    ROBUST_BYTES,
    ROBUST_MESSAGES,
    SEED,
    SEGMENT_BITS,
)
from full_exp.prepare_detections import prepare_all
from stego_exp.geo_atoms import attach_atoms, load_schemes
from stego_exp.lazy_images import LazyImages
from stego_exp.paper_attacks import (
    ATTACK_DISPLAY,
    PAPER_ATTACK_NAMES,
    apply_paper_attack,
)
from stego_exp.pipeline import CoverlessSystem, recover_with_recall, segment_accuracy
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


def eval_robustness_table_ix(system, path_to_img, n_messages, msg_bytes, seed=49):
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
    for atk in tqdm(PAPER_ATTACK_NAMES, desc="TableIX", mininterval=5.0):
        t0 = time.time()
        accs, r1s, rpools = [], [], []
        for msg, res in hides:
            attacked = [apply_paper_attack(path_to_img[p], atk) for p in res.stego_paths]
            pred, r1, rpool = recover_with_recall(
                system, attacked, res.aux, res.pad_zeros, path_to_img
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
        # checkpoint after each attack
        ckpt = RESULTS / "_robust_ckpt_tmp.json"
        ckpt.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out["_avg"] = {
        "ACC": float(np.mean([out[a]["ACC"] for a in PAPER_ATTACK_NAMES])),
        "R@1": float(np.mean([out[a]["R@1"] for a in PAPER_ATTACK_NAMES])),
        "R@pool": float(np.mean([out[a]["R@pool"] for a in PAPER_ATTACK_NAMES])),
    }
    geo = PAPER_ATTACK_NAMES[:13]
    nongeo = PAPER_ATTACK_NAMES[13:]
    out["_avg_geometry"] = {
        "ACC": float(np.mean([out[a]["ACC"] for a in geo])),
        "R@1": float(np.mean([out[a]["R@1"] for a in geo])),
    }
    out["_avg_nongeometry"] = {
        "ACC": float(np.mean([out[a]["ACC"] for a in nongeo])),
        "R@1": float(np.mean([out[a]["R@1"] for a in nongeo])),
    }
    return out


def run_one_dataset(
    name: str,
    pred_cache: Path,
    scheme_json: Path,
    *,
    capacity_trials: int = CAPACITY_TRIALS,
    robust_messages: int = ROBUST_MESSAGES,
    robust_bytes: int = ROBUST_BYTES,
):
    print(f"\n########## {name} ##########")
    preds = json.loads(pred_cache.read_text(encoding="utf-8"))
    schemes = load_schemes(scheme_json, name)

    det = _filter_proposed(preds, conf=0.5)
    det = attach_atoms(det, schemes, use_geo=True)
    paths = sorted(p for p in det if Path(p).exists())
    det = {p: det[p] for p in paths}
    print(f"usable images with objs: {len(paths)}")

    lazy = LazyImages(max_cache=128)
    lazy.add_paths(paths)

    bp = bank_path_for(name, len(paths))
    bank = build_bank_chunked(paths, bp, N_CLUSTERS[name])

    system = CoverlessSystem(
        detections=det,
        path_to_img=lazy,
        bank=bank,
        segment_bits=SEGMENT_BITS,
        area_ratio_thresh=0.0,
        max_objects=MAX_OBJECTS,
        use_attributes=False,
        retrieve_topk=RETRIEVE_TOPK,
    )

    t0 = time.time()
    stats = object_stats(system)
    print("stats", stats)
    cap = eval_capacity(system, MSG_LENGTHS, capacity_trials, seed=SEED)
    print("capacity", json.dumps(cap, indent=2))
    rob = eval_robustness_table_ix(
        system, lazy, robust_messages, robust_bytes, seed=SEED + 7
    )
    print("robust avg", rob.get("_avg"), "geo", rob.get("_avg_geometry"))

    return {
        "dataset": name,
        "n_images": len(paths),
        "segment_bits": SEGMENT_BITS,
        "area_thresh": AREA_THRESH,
        "max_objects": MAX_OBJECTS,
        "n_clusters": N_CLUSTERS[name],
        "stats": stats,
        "capacity": cap,
        "robustness": rob,
        "seconds": time.time() - t0,
    }


def write_report(all_res: dict, path: Path):
    lines = [
        "# Full-Dataset Proposed Method Results",
        "",
        "Protocol: proposed method (geo subclasses + hybrid retrieval).",
        "",
        "- Datasets: VOC2007 train+test, VOC2012 full, COCO2017 train+val",
        "- Detector: YOLOv8m @ conf=0.5",
        "- Codebook: adaptive geometric subclasses, n=6",
        "- Object select: area≥1.5%, max 8",
        "- Retrieval: global descriptor recall + ORB/SSIM hybrid",
        f"- Capacity: avg #stego for {list(MSG_LENGTHS)} bytes × {CAPACITY_TRIALS} trials",
        f"- Robustness: Table IX attacks, msg={ROBUST_BYTES}B × {ROBUST_MESSAGES} trials",
        "",
    ]
    for name, res in all_res.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"images={res['n_images']}, n={res['segment_bits']}, "
            f"atoms={res['stats']['codebook_size_atoms']}, "
            f"objs/img={res['stats']['avg_objects']:.2f}, k={res['n_clusters']}"
        )
        lines.append("")
        lines.append("### Capacity (avg stego ↓)")
        lines.append("")
        lines.append("| 10B | 100B | 1KB | 2KB |")
        lines.append("|---:|---:|---:|---:|")

        def fmt(L):
            row = res["capacity"].get(str(L), {})
            v = row.get("avg_stego")
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


def run_all():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    print("=== 1/3 prepare detections ===")
    det_paths = prepare_all()

    print("=== 2/3 build geometric schemes ===")
    scheme_json = RESULTS / "geo_subclass_full.json"
    if not scheme_json.exists():
        scheme_json = build_all_schemes(det_paths)
    else:
        print(f"reuse {scheme_json}")

    print("=== 3/3 proposed capacity + robustness ===")
    all_res = {}
    partial = RESULTS / "full_proposed_results.partial.json"
    if partial.exists():
        all_res = json.loads(partial.read_text(encoding="utf-8"))
        print(f"resume partial datasets: {list(all_res)}")

    for name, cache in det_paths.items():
        if name in all_res:
            print(f"skip done {name}")
            continue
        res = run_one_dataset(name, cache, scheme_json)
        all_res[name] = res
        partial.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")

    out_json = RESULTS / "full_proposed_results.json"
    out_md = RESULTS / "full_proposed_results.md"
    out_json.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(all_res, out_md)
    print("Wrote", out_md)
    return all_res


if __name__ == "__main__":
    run_all()
