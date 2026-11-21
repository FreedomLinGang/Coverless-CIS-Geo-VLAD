# -*- coding: utf-8 -*-
"""Regenerate fig03 (uniform vs adaptive, exact numbers) and fig_marginal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(r"D:\a_python_codes\论文5\our_code")
sys.path.insert(0, str(ROOT))

FIG = Path(
    r"D:\a_python_codes\论文5\论文投稿打包_VLAD几何细分类_20260825\01_论文与图表\figures"
)
FIG.mkdir(parents=True, exist_ok=True)

GEO = json.loads(
    Path(
        r"D:\a_python_codes\论文5\our_code\full_exp\results\geo_subclass_full.json"
    ).read_text(encoding="utf-8")
)
UNI = json.loads(
    Path(
        r"D:\a_python_codes\论文5\论文投稿打包_VLAD几何细分类_20260825"
        r"\02_实验结果\geo_refine\uniform_vs_adaptive.json"
    ).read_text(encoding="utf-8")
)

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 10,
})
BLUE, GRAY, GREEN, ORANGE = "#2E5A88", "#8C94A1", "#2E7D4F", "#C47A2C"

DS = ["VOC2007", "VOC2012", "COCO2017"]
X = np.arange(3)
W = 0.35


def fig_balance():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    uni_raw = [GEO[d]["reference_uniform"]["n_subclasses_raw"] for d in DS]
    ada_raw = [GEO[d]["adaptive_summary"]["n_subclasses_raw"] for d in DS]
    uni_sparse = [GEO[d]["reference_uniform"]["sparse_ratio"] * 100 for d in DS]
    ada_sparse = [
        GEO[d]["adaptive_summary"]["n_sparse_raw"] / GEO[d]["adaptive_summary"]["n_subclasses_raw"] * 100
        for d in DS
    ]
    uni_usable = [UNI[d]["uniform_usable"] for d in DS]
    ada_usable = [GEO[d]["adaptive_summary"]["n_subclasses_merged"] for d in DS]
    uni_cv = [UNI[d]["uniform_mean_cv"] for d in DS]
    ada_cv = [GEO[d]["adaptive_summary"]["mean_balance_cv"] for d in DS]

    ax = axes[0]
    ax.bar(X - W / 2, uni_raw, W, label="固定4×8×4", color=GRAY)
    ax.bar(X + W / 2, ada_raw, W, label="自适应方案", color=BLUE)
    for i, (u, a) in enumerate(zip(uni_raw, ada_raw)):
        ax.text(i - W / 2, u + 40, f"{u}", ha="center", fontsize=8)
        ax.text(i + W / 2, a + 40, f"{a}", ha="center", fontsize=8)
    ax.set_xticks(X)
    ax.set_xticklabels(DS)
    ax.set_ylabel("原始子类数")
    ax.set_title("(a) 网格切分产生的原始子类数")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(X - W / 2, uni_usable, W, label="固定4×8×4", color=GRAY)
    ax.bar(X + W / 2, ada_usable, W, label="自适应方案", color=GREEN)
    for i, (u, a) in enumerate(zip(uni_usable, ada_usable)):
        ax.text(i - W / 2, u + 30, f"{u}", ha="center", fontsize=8)
        ax.text(i + W / 2, a + 30, f"{a}", ha="center", fontsize=8)
    ax.set_xticks(X)
    ax.set_xticklabels(DS)
    ax.set_ylabel("合并后可用子类数")
    ax.set_title("(b) 最小容量合并后可用子类数")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(X - W / 2, uni_sparse, W, label="固定4×8×4", color=GRAY)
    ax.bar(X + W / 2, ada_sparse, W, label="自适应方案", color=ORANGE)
    for i, (u, a) in enumerate(zip(uni_sparse, ada_sparse)):
        ax.text(i - W / 2, u + 2, f"{u:.1f}%", ha="center", fontsize=8)
        ax.text(i + W / 2, a + 2, f"{a:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(X)
    ax.set_xticklabels(DS)
    ax.set_ylabel("稀疏子类占比 (%)")
    ax.set_title("(c) 低于最小容量的稀疏子类占比")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("均匀4×8×4网格 vs 按类自适应划分（YOLOv8m @ conf=0.5）",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig03_balance.png", bbox_inches="tight", facecolor="white")
    print("saved fig03_balance.png")


def fig_marginal():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for ds_panels, ax, title, color in [
        ("marginals", axes[0], "(a) 对角线角分布", BLUE),
        (None, axes[1], "(b) 方位角分布", GREEN),
        (None, axes[2], "(c) 归一化距离分布（分位数）", ORANGE),
    ]:
        pass
    m = GEO["VOC2007"]["marginals"]
    d = m["diag_4x22.5"]
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    axes[0].bar(range(4), vals, color=BLUE)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(["0-22.5", "22.5-45", "45-67.5", "67.5-90"], fontsize=8)
    axes[0].set_xlabel("对角线角 θd (°)")
    axes[0].set_ylabel("检测框数")
    axes[0].set_title("(a) 对角线角分布", fontsize=10)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 150, f"{v / sum(vals) * 100:.0f}%", ha="center", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    s = m["spatial_8x45"]
    svals = [s[k] for k in s.keys()]
    rot = svals[4:] + svals[:4]
    axes[1].bar(range(8), rot, color=GREEN)
    axes[1].set_xticks(range(8))
    axes[1].set_xticklabels(["-135", "-90", "-45", "0", "45", "90", "135", "180"], fontsize=7)
    axes[1].set_xlabel("方位角 θs (°，180=图像下方)")
    axes[1].set_title("(b) 方位角分布", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    r = m["dist_quartile"]
    rvals = [r[k] for k in r.keys()]
    axes[2].bar(range(4), rvals, color=ORANGE)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=8)
    axes[2].set_xlabel("分位数区间")
    axes[2].set_title("(c) 距离比分布（类内分位数）", fontsize=10)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("VOC2007 检测框几何特征边际分布（YOLOv8m，28560个检测框）",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig_marginal.png", bbox_inches="tight", facecolor="white")
    print("saved fig_marginal.png")


if __name__ == "__main__":
    fig_balance()
    fig_marginal()
