# -*- coding: utf-8 -*-
"""Codebook packing statistics + Example-1 style group dump (VOC2007 / COCO2017).

Mirrors reference Fig. 5: geometric subclasses packed into 2^n codewords with
least-loaded-first balancing; report per-code image coverage stats and example
groups shared by subclasses from different classes.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\a_python_codes\论文5\our_code")
sys.path.insert(0, str(ROOT))

from full_exp_cluster_sift.config import AREA_THRESH, DET_CACHES, SCHEME_JSON  # noqa: E402
from stego_exp.codebook import build_codebook  # noqa: E402
from stego_exp.geo_atoms import attach_atoms, load_schemes  # noqa: E402

OUT = Path(
    r"D:\a_python_codes\论文5\论文投稿打包_VLAD几何细分类_20260825"
    r"\02_实验结果\geo_refine\codebook_pack_stats.md"
)


def filter_dets(preds, max_obj=8):
    out = {}
    for path, dets in preds.items():
        rows = [d for d in dets if float(d.get("score", 0)) >= 0.5]
        rows = sorted(rows, key=lambda d: -float(d.get("area_ratio", 0)))
        kept = [d for d in rows if float(d.get("area_ratio", 0)) >= AREA_THRESH]
        rows = kept if kept else rows[:1]
        rows = rows[:max_obj]
        if rows:
            out[path] = rows
    return out


def analyze(name, segment_bits=6):
    print(f"\n==== {name} ====", flush=True)
    preds = json.loads(DET_CACHES[name].read_text(encoding="utf-8"))
    schemes = load_schemes(SCHEME_JSON, name)
    det = attach_atoms(filter_dets(preds), schemes, use_geo=True)
    atoms = []
    for path, objs in det.items():
        for o in objs:
            atoms.append(o["atom"])
    n_atoms_unique = len(set(atoms))
    print(f"usable images={len(det)} atoms(instances)={len(atoms)} unique={n_atoms_unique}")
    codebook = build_codebook(det, segment_bits, use_attributes=False, image_loader=None)
    n_codes = 2**segment_bits
    cover = defaultdict(int)
    per_atom = {}
    for path, objs in det.items():
        used = set(o["atom"] for o in objs)
        for a in used:
            cover[codebook[a]] += 1
            per_atom.setdefault(a, codebook[a])
    cov = np.array([cover.get(format(g, f"0{segment_bits}b"), 0) for g in range(n_codes)],
                   dtype=np.float64)
    print(f"codes={n_codes} empty_codes={(cov == 0).sum()} "
          f"min={cov.min():.0f} mean={cov.mean():.1f} max={cov.max():.0f} "
          f"min/max={cov.min() / max(cov.max(), 1):.3f} std={cov.std():.1f}")
    # groups (atoms per code) stats
    group_of = defaultdict(list)
    for a, g in codebook.items():
        group_of[g].append(a)
    sizes = [len(v) for v in group_of.values()]
    print(f"atoms/code: min={min(sizes)} mean={np.mean(sizes):.2f} max={max(sizes)}")
    # example groups with subclasses from >= 2 different classes
    cap = 6 if n_atoms_unique < 500 else 40
    examples = []
    for g, ats in sorted(group_of.items()):
        classes = [a.split("|")[0] for a in ats]
        if len(set(classes)) >= 2 and len(ats) <= cap:
            examples.append((g, ats))
    print("example shared groups (>=2 classes):")
    for g, ats in examples[:6]:
        print(f"  group {g} <- {ats}")

    # strawman: sequential assignment by label order (no balancing) coverage stats
    atoms_sorted = sorted(codebook.keys())
    seq_cov = defaultdict(int)
    for a, pcode in zip(atoms_sorted, [format(i % n_codes, f"0{segment_bits}b")
                                       for i in range(len(atoms_sorted))]):
        seq_cov[pcode] += 0
    # recompute per-atom image coverage
    atom_img_cov: dict[str, int] = defaultdict(int)
    seen = set()
    for path, objs in det.items():
        used = {o["atom"] for o in objs}
        for a in used:
            atom_img_cov[a] += 1
    for a in atoms_sorted:
        pcode = format(atoms_sorted.index(a) % n_codes, f"0{segment_bits}b")
        seq_cov[pcode] += atom_img_cov[a]
    scov = np.array([seq_cov.get(format(g, f"0{segment_bits}b"), 0) for g in range(n_codes)],
                    dtype=np.float64)
    print(f"[strawman sequential] min={scov.min():.0f} mean={scov.mean():.1f} max={scov.max():.0f} "
          f"min/max={scov.min() / max(scov.max(), 1):.3f} std={scov.std():.1f}")
    return {
        "dataset": name,
        "usable_images": len(det),
        "unique_atoms": n_atoms_unique,
        "codes": n_codes,
        "empty_codes": int((cov == 0).sum()),
        "cov_min": float(cov.min()),
        "cov_mean": float(cov.mean()),
        "cov_max": float(cov.max()),
        "cov_minmax_ratio": float(cov.min() / max(cov.max(), 1.0)),
        "cov_std": float(cov.std()),
        "atoms_per_code_min": min(sizes),
        "atoms_per_code_mean": float(np.mean(sizes)),
        "atoms_per_code_max": max(sizes),
        "seq_cov_min": float(scov.min()),
        "seq_cov_mean": float(scov.mean()),
        "seq_cov_max": float(scov.max()),
        "seq_cov_minmax_ratio": float(scov.min() / max(scov.max(), 1.0)),
        "seq_cov_std": float(scov.std()),
        "examples": [{"group": g, "atoms": ats} for g, ats in examples[:5]],
    }


def main():
    rows = [analyze("VOC2007", 6)]
    rows.append(analyze("COCO2017", 6))
    lines = ["# 码本打包统计（n=6，最小负载优先均衡）", ""]
    for r in rows:
        lines.append(f"## {r['dataset']}")
        lines.append(f"- 可用图像数: {r['usable_images']}, 几何细分类数: {r['unique_atoms']}, "
                     f"码字数: {r['codes']}（空码字 {r['empty_codes']}）")
        lines.append(f"- 每个码字的可映射图像数: min={r['cov_min']:.0f}, mean={r['cov_mean']:.1f}, "
                     f"max={r['cov_max']:.0f}, min/max={r['cov_minmax_ratio']:.3f}, std={r['cov_std']:.1f}")
        lines.append(f"- 每码字子类数: min={r['atoms_per_code_min']}, mean={r['atoms_per_code_mean']:.2f}, "
                     f"max={r['atoms_per_code_max']}")
        lines.append("- 跨类共享码字示例:")
        for e in r["examples"]:
            lines.append(f"  - 码字 {e['group']} <- {e['atoms']}")
        lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
