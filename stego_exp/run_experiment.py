"""Run capacity + robustness for the proposed method."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .attacks import DEFAULT_ATTACKS, attack_image
from .config import CACHE_DIR, DATASETS, DEFAULT_CFG, RESULT_DIR, ExpConfig
from .detect import build_detection_cache, list_images, load_detection_cache
from .pipeline import CoverlessSystem, segment_accuracy
from .retrieval import FeatureBank
from .voc_gt import build_voc_gt_detections


def load_images(paths: list[Path]) -> dict[str, Image.Image]:
    out = {}
    for p in tqdm(paths, desc="Loading images"):
        out[str(p)] = Image.open(p).convert("RGB")
    return out


def make_system(cfg: ExpConfig, detections, path_to_img, bank) -> CoverlessSystem:
    return CoverlessSystem(
        detections=detections,
        path_to_img=path_to_img,
        bank=bank,
        segment_bits=cfg.segment_bits,
        area_ratio_thresh=cfg.area_ratio_thresh,
        max_objects=cfg.max_objects_per_image,
        use_attributes=cfg.use_attributes,
        retrieve_topk=cfg.retrieve_topk,
    )


def eval_capacity(system: CoverlessSystem, lengths, trials, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    rows = {}
    for L in lengths:
        counts = []
        fails = 0
        for t in range(trials):
            msg = bytes(rng.integers(0, 256, size=L, dtype=np.uint8).tolist())
            res = system.hide(msg)
            if res is None:
                fails += 1
            else:
                counts.append(res.n_stego)
        rows[str(L)] = {
            "success_rate": (trials - fails) / trials,
            "avg_stego": float(np.mean(counts)) if counts else None,
            "std_stego": float(np.std(counts)) if counts else None,
            "fails": fails,
            "trials": trials,
        }
    return rows


def eval_robustness(system: CoverlessSystem, path_to_img, attacks, n_messages, msg_bytes, seed=1) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    hides = []
    for _ in range(n_messages):
        msg = bytes(rng.integers(0, 256, size=msg_bytes, dtype=np.uint8).tolist())
        res = system.hide(msg)
        if res is not None:
            hides.append((msg, res))
    if not hides:
        return {"error": "no successful hides"}

    for atk in tqdm(attacks, desc="Attacks[proposed]"):
        accs, hits = [], []
        for msg, res in hides:
            attacked = [attack_image(path_to_img[p], atk) for p in res.stego_paths]
            pred, hit = system.recover_from_paths(attacked, res.aux, res.pad_zeros)
            accs.append(segment_accuracy(res.message_bits, pred, system.segment_bits, res.pad_zeros))
            hits.append(hit)
        out[atk] = {
            "ACC": float(np.mean(accs)),
            "R@1": float(np.mean(hits)),
            "n": len(hides),
        }
    out["_avg"] = {
        "ACC": float(np.mean([v["ACC"] for k, v in out.items() if not k.startswith("_")])),
        "R@1": float(np.mean([v["R@1"] for k, v in out.items() if not k.startswith("_")])),
    }
    geo = [k for k in out if k.startswith(("rotate", "translate", "scale", "crop"))]
    if geo:
        out["_avg_geometry"] = {
            "ACC": float(np.mean([out[k]["ACC"] for k in geo])),
            "R@1": float(np.mean([out[k]["R@1"] for k in geo])),
        }
    return out


def object_stats(system: CoverlessSystem) -> dict:
    counts = []
    for p in system.path_to_img:
        counts.append(len(system._selected(p)))
    return {
        "avg_objects": float(np.mean(counts)) if counts else 0,
        "max_objects": int(np.max(counts)) if counts else 0,
        "min_objects": int(np.min(counts)) if counts else 0,
        "images_with_ge4": int(np.sum(np.array(counts) >= 4)),
        "images_with_ge5": int(np.sum(np.array(counts) >= 5)),
        "codebook_size_atoms": len(system.codebook),
        "unique_codes_used": len(set(system.codebook.values())),
    }


def run(cfg: ExpConfig | None = None):
    cfg = cfg or DEFAULT_CFG
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    folder = DATASETS[cfg.dataset]
    assert folder.exists(), f"Missing dataset folder: {folder}"
    paths = list_images(folder, cfg.max_images, cfg.seed)
    print(f"Dataset={cfg.dataset} images={len(paths)} folder={folder}")

    if cfg.dataset == "voc2012":
        ann_dir = Path(r"D:\datasets\VOC2012\Annotations")
        cache_path = CACHE_DIR / f"det_voc_gt_n{len(paths)}.json"
        if cache_path.exists():
            detections = load_detection_cache(cache_path)
        else:
            detections = build_voc_gt_detections(folder, ann_dir, paths)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(detections, f)
            print(f"Saved VOC-GT detections: {cache_path}")
    else:
        cache = build_detection_cache(cfg, paths)
        detections = load_detection_cache(cache)

    paths = [p for p in paths if detections.get(str(p))]
    path_to_img = load_images(paths)
    detections = {str(p): detections[str(p)] for p in paths}
    print(f"Usable images with objects: {len(paths)}")

    print("Building feature bank + clusters...")
    bank = FeatureBank(device=cfg.device)
    bank.build(path_to_img, n_clusters=cfg.n_clusters)

    t0 = time.time()
    system = make_system(cfg, detections, path_to_img, bank)
    stats = object_stats(system)
    print("object/codebook stats:", stats)
    cap = eval_capacity(system, cfg.message_lengths, cfg.capacity_trials, seed=cfg.seed)
    print("capacity:", json.dumps(cap, indent=2))
    rob = eval_robustness(
        system,
        path_to_img,
        DEFAULT_ATTACKS,
        n_messages=cfg.robustness_messages,
        msg_bytes=cfg.message_bytes_for_robust,
        seed=cfg.seed + 7,
    )
    print("robustness avg:", rob.get("_avg"), "geo:", rob.get("_avg_geometry"))

    all_results = {
        "config": cfg.__dict__,
        "n_images": len(paths),
        "method": "proposed",
        "stats": stats,
        "capacity": cap,
        "robustness": rob,
        "seconds": time.time() - t0,
    }

    out_path = RESULT_DIR / f"proposed_{cfg.dataset}_n{len(paths)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    md = RESULT_DIR / f"proposed_{cfg.dataset}_n{len(paths)}.md"
    write_markdown(all_results, md)
    print(f"Saved: {md}")
    return all_results


def write_markdown(results: dict, path: Path):
    lines = []
    lines.append("# Proposed Method Results\n")
    lines.append(f"- images: **{results['n_images']}**")
    lines.append(f"- dataset: `{results['config']['dataset']}`")
    lines.append(f"- segment_bits: {results['config']['segment_bits']}\n")

    lines.append("## Capacity (avg stego images, lower is better)\n")
    lines.append("| 10B | 100B | 1000B | avg objects/img |")
    lines.append("|---:|---:|---:|---:|")
    cap = results["capacity"]

    def fmt(L):
        v = cap[str(L)]["avg_stego"]
        sr = cap[str(L)]["success_rate"]
        if v is None:
            return f"fail({sr:.0%})"
        return f"{v:.1f}"

    lines.append(
        f"| {fmt(10)} | {fmt(100)} | {fmt(1000)} | {results['stats']['avg_objects']:.2f} |"
    )

    lines.append("\n## Robustness average\n")
    lines.append("| ACC | R@1 | Geometry ACC | Geometry R@1 |")
    lines.append("|---:|---:|---:|---:|")
    rob = results["robustness"]
    if isinstance(rob, dict) and "error" not in rob and rob.get("_avg"):
        avg = rob.get("_avg", {})
        geo = rob.get("_avg_geometry", {})
        lines.append(
            f"| {avg.get('ACC',0):.3f} | {avg.get('R@1',0):.3f} | "
            f"{geo.get('ACC',0):.3f} | {geo.get('R@1',0):.3f} |"
        )
    else:
        lines.append("| n/a | n/a | n/a | n/a |")

    lines.append("\n## Per-attack ACC / R@1\n")
    if isinstance(rob, dict) and "error" not in rob:
        attacks = [k for k in rob.keys() if not k.startswith("_") and isinstance(rob.get(k), dict)]
        lines.append("| Attack | ACC | R@1 |")
        lines.append("|---|---:|---:|")
        for a in attacks:
            lines.append(f"| {a} | {rob[a]['ACC']:.3f} | {rob[a]['R@1']:.3f} |")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run(DEFAULT_CFG)
