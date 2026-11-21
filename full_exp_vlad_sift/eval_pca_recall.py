# -*- coding: utf-8 -*-
"""Compare full-dim VLAD vs float16 / PCA shortlist recall."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from full_exp_vlad_sift.build_vlad import load_or_build_vlad
from full_exp_vlad_sift.config import SHORTLIST
from no_aux_hash.gallery import load_gallery
from no_aux_hash.hashes import topk_indices
from no_aux_hash.he_index import vlad_vector
from no_aux_hash.rich_features import detect_sift
from stego_exp.paper_attacks import apply_paper_attack


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def scores_f16(g16, q32, bs=4096):
    n = g16.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, bs):
        out[i : i + bs] = np.asarray(g16[i : i + bs], dtype=np.float32) @ q32
    return out


def run(name="VOC2012", n_queries=80, seed=0):
    paths, images, *_ = load_gallery(name)
    g_full, centers = load_or_build_vlad(name, paths, images)
    g_full = np.array(g_full, dtype=np.float32, copy=True)
    g16 = g_full.astype(np.float16)
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(paths), size=min(n_queries, len(paths)), replace=False)

    def fit_pca(dim):
        sample = rng.choice(len(paths), size=min(8000, len(paths)), replace=False)
        m = PCA(n_components=dim, random_state=0)
        m.fit(g_full[sample])
        mean, W = m.mean_.astype(np.float32), m.components_.T.astype(np.float32)
        gp = _l2((g_full - mean) @ W)
        print(f"  PCA-{dim} explained={float(m.explained_variance_ratio_.sum()):.3f}")
        return mean, W, gp

    p256 = fit_pca(256)
    p1024 = fit_pca(1024)
    print(f"{name} n={len(paths)}")

    attacks = ["edge_crop_20", "rotate_50", "hist_eq"]
    for atk in attacks:
        stats = {k: [] for k in ("full40", "f1640", "p256_40", "p1024_40")}
        for i in tqdm(q_idx, desc=atk):
            img = apply_paper_attack(images[paths[int(i)]], atk)
            _, desc = detect_sift(img)
            q = vlad_vector(desc, centers)
            qn = q / (np.linalg.norm(q) + 1e-8)
            gt = int(i)
            stats["full40"].append(gt in topk_indices(g_full @ qn, SHORTLIST, True))
            stats["f1640"].append(gt in topk_indices(scores_f16(g16, qn), SHORTLIST, True))
            qp = _l2((q - p256[0]) @ p256[1])
            stats["p256_40"].append(gt in topk_indices(p256[2] @ qp, SHORTLIST, True))
            qp = _l2((q - p1024[0]) @ p1024[1])
            stats["p1024_40"].append(gt in topk_indices(p1024[2] @ qp, SHORTLIST, True))
        print(
            f"  {atk} R@40  full={np.mean(stats['full40']):.3f}  "
            f"f16={np.mean(stats['f1640']):.3f}  "
            f"PCA256={np.mean(stats['p256_40']):.3f}  "
            f"PCA1024={np.mean(stats['p1024_40']):.3f}"
        )


if __name__ == "__main__":
    run()
