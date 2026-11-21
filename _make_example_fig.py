# -*- coding: utf-8 -*-
"""Generate example figure: a real VOC2007 image with YOLOv8m boxes sorted by area,
annotated with geometric subclass IDs (mirrors reference Fig. 6/7 examples)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\a_python_codes\论文5\our_code")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from PIL import Image  # noqa: E402

from full_exp_cluster_sift.config import AREA_THRESH, DET_CACHES, SCHEME_JSON  # noqa: E402
from stego_exp.geo_atoms import attach_atoms, load_schemes  # noqa: E402

OUT = Path(
    r"D:\a_python_codes\论文5\论文投稿打包_VLAD几何细分类_20260825\01_论文与图表\figures"
)
OUT.mkdir(parents=True, exist_ok=True)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f"]


def pick_dets(preds, max_obj=8):
    rows = [d for d in preds if float(d.get("score", 0)) >= 0.5]
    rows = sorted(rows, key=lambda d: -float(d.get("area_ratio", 0)))
    kept = [d for d in rows if float(d.get("area_ratio", 0)) >= AREA_THRESH]
    rows = kept if kept else rows[:1]
    return rows[:max_obj]


def main():
    preds = json.loads(DET_CACHES["VOC2007"].read_text(encoding="utf-8"))
    schemes = load_schemes(SCHEME_JSON, "VOC2007")
    # candidate images with at least 4 kept detections; prefer diverse classes
    def score(p):
        ds = pick_dets(preds[p])
        ncls = len({d["label"] for d in ds})
        return len(ds) * 10 + ncls
    cands = [p for p, raw in preds.items() if len(pick_dets(raw)) >= 4]
    cands.sort(key=score, reverse=True)
    path = cands[0]
    # attach atoms on the filtered set so atom IDs are consistent with the pipeline
    det0 = {p: pick_dets(raw) for p, raw in preds.items()}
    det0 = attach_atoms(det0, schemes, use_geo=True)
    dets = det0[path]
    print("picked", path, "objs", len(dets),
          "classes", sorted({d["label"] for d in dets}))

    img = Image.open(path).convert("RGB")
    W, H = img.size
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    ax.imshow(img)
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d["box"]
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False,
                edgecolor=COLORS[i % len(COLORS)], linewidth=3.2,
            )
        )
        atom = d.get("atom", d["label"])
        txt = f"#{i + 1}  {atom}  conf={d['score']:.2f}"
        # label with white background
        ax.text(
            x1, max(4, y1 - 8), txt, fontsize=11, color="black", weight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor=COLORS[i % len(COLORS)],
                      linewidth=1.2, pad=1.5),
            va="bottom", ha="left",
        )
        # order arrow
        if i < len(dets) - 1:
            xc = max(x1, 4)
            ax.annotate(
                "", xy=(x1 + 2, y1), xytext=(x1 + 2, y1 + 10),
                arrowprops=dict(arrowstyle="-", color=COLORS[i % len(COLORS)]),
            )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "VOC2007 real image: top-8 boxes by area ratio (YOLOv8m conf>=0.5, area>=1.5%)  ->  geometric subclasses",
        fontsize=13,
    )
    for s in ("top", "bottom", "left", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = OUT / "fig_example_detections.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print("wrote", p)
    # also dump order info for the paper example
    for i, d in enumerate(dets):
        print(i + 1, d["label"], "| atom:", d.get("atom"), "| area_ratio:",
              round(float(d["area_ratio"]), 4), "| conf:", round(float(d["score"]), 3))


if __name__ == "__main__":
    main()
