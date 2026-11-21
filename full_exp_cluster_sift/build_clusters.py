# -*- coding: utf-8 -*-
"""SIFT mean-descriptor features + flat k-means (paper-3 feature, flat like paper-1)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from full_exp_cluster_sift.cluster_index import ClusterIndex
from full_exp_cluster_sift.config import CACHE, N_CLUSTERS, SIFT_SIZE, SIFT_T


def sift_mean_feature(img: Image.Image, size: int = SIFT_SIZE) -> np.ndarray:
    """Normalized average of SIFT descriptors (参考论文3 CID feature, flat use)."""
    gray = cv2.cvtColor(
        np.asarray(img.convert("RGB").resize((size, size))),
        cv2.COLOR_RGB2GRAY,
    )
    sift = cv2.SIFT_create()
    _, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        return np.zeros(128, dtype=np.float32)
    feat = desc.astype(np.float32).mean(axis=0)
    nrm = float(np.linalg.norm(feat)) + 1e-8
    return (feat / nrm).astype(np.float32)


def load_or_build_sift_feats(name: str, paths: list[str], path_to_img, chunk: int = 256) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"siftmean_{name}_n{len(paths)}.npz"
    if out.exists():
        data = np.load(out, allow_pickle=True)
        sp = data["paths"].tolist()
        feats = data["feats"].astype(np.float32)
        if sp == paths:
            print(f"load SIFT-mean feats {out.name}")
            return feats
        idx = {p: i for i, p in enumerate(sp)}
        if all(p in idx for p in paths):
            print(f"reorder SIFT-mean feats {out.name}")
            return np.stack([feats[idx[p]] for p in paths], axis=0)

    print(f"extract SIFT-mean feats n={len(paths)}")
    feats_list: list[np.ndarray] = []
    done: list[str] = []
    # resume partial
    part = out.with_suffix(".partial.npz")
    if part.exists():
        pdata = np.load(part, allow_pickle=True)
        done = pdata["paths"].tolist()
        feats_list = [pdata["feats"][i] for i in range(len(done))]
        print(f"  resume {len(done)}")
    done_set = set(done)
    todo = [p for p in paths if p not in done_set]
    for i in tqdm(range(0, len(todo), chunk), desc="SIFT-mean"):
        batch = todo[i : i + chunk]
        for p in batch:
            img = path_to_img[p]
            if not isinstance(img, Image.Image):
                img = Image.open(p).convert("RGB")
            feats_list.append(sift_mean_feature(img))
            done.append(p)
        if (i // chunk) % 10 == 0 or i + chunk >= len(todo):
            arr = np.stack(feats_list, axis=0).astype(np.float32)
            np.savez_compressed(part, paths=np.array(done, dtype=object), feats=arr)
    # align to paths order
    idx = {p: i for i, p in enumerate(done)}
    feats = np.stack([feats_list[idx[p]] for p in paths], axis=0).astype(np.float32)
    np.savez_compressed(out, paths=np.array(paths, dtype=object), feats=feats)
    if part.exists():
        part.unlink()
    print(f"saved {out}")
    return feats


def build_cluster_index(name: str, paths: list[str], path_to_img) -> ClusterIndex:
    CACHE.mkdir(parents=True, exist_ok=True)
    k = N_CLUSTERS[name]
    out = CACHE / f"clusters_sift_{name}_n{len(paths)}_k{k}.npz"
    if out.exists():
        data = np.load(out, allow_pickle=True)
        c_paths = data["paths"].tolist()
        labels = data["labels"]
        feats = data["feats"]
        if c_paths == paths:
            print(f"load clusters {out.name}")
            return ClusterIndex(c_paths, {p: int(labels[i]) for i, p in enumerate(c_paths)}, feats)

    feats = load_or_build_sift_feats(name, paths, path_to_img)
    kk = min(k, max(2, len(paths) // 4))
    print(f"k-means++ on SIFT-mean: n={len(paths)} k={kk}")
    km = MiniBatchKMeans(
        n_clusters=kk,
        init="k-means++",
        random_state=42,
        batch_size=2048,
        n_init=3,
        max_iter=200,
    )
    labels = km.fit_predict(feats)
    np.savez_compressed(
        out,
        paths=np.array(paths, dtype=object),
        labels=labels.astype(np.int32),
        feats=feats,
        centers=km.cluster_centers_,
    )
    sizes = np.bincount(labels)
    print(f"saved {out}; size min={sizes.min()} med={int(np.median(sizes))} max={sizes.max()}")
    return ClusterIndex(paths, {p: int(labels[i]) for i, p in enumerate(paths)}, feats)


def sift_inlier_count(a: Image.Image, b: Image.Image, t: float = SIFT_T) -> int:
    from full_exp_sift.retrieval_sift import SIFTCache, sift_inlier_count_from_feat

    # lightweight one-off (caller should pass cache for batch)
    cache = SIFTCache(max_items=8)
    ka, da = cache.get("q", a)
    kb, db = cache.get("g", b)
    return sift_inlier_count_from_feat(ka, da, kb, db, t=t)


def match_sift_in_cluster(
    query: Image.Image,
    candidates: list[str],
    path_to_img,
    cache,
    index: ClusterIndex | None = None,
    shortlist: int = 40,
) -> str:
    """Within-cluster SIFT-J. If cluster is large, shortlist by SIFT-mean first."""
    if not candidates:
        return ""
    from full_exp_sift.retrieval_sift import sift_inlier_count_from_feat

    cands = list(candidates)
    if index is not None and index.feats is not None and len(cands) > shortlist:
        # paper-3 style mean descriptor as cheap within-cluster prefilter
        qf = sift_mean_feature(query)
        scored = []
        for p in cands:
            i = index.index[p]
            scored.append((float(qf @ index.feats[i]), p))
        scored.sort(reverse=True)
        cands = [p for _, p in scored[:shortlist]]

    qk, qd = cache.get("__query__", query)
    best_p, best_j = cands[0], -1
    for p in cands:
        kb, db = cache.get(p, path_to_img[p])
        j = sift_inlier_count_from_feat(qk, qd, kb, db)
        if j > best_j:
            best_j, best_p = j, p
    cache._store.pop("__query__", None)
    return best_p
