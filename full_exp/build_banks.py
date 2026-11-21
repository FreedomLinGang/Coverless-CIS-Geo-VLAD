# -*- coding: utf-8 -*-
"""Build / resume feature banks for full datasets (chunked embedding)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from full_exp.config import CACHE, N_CLUSTERS
from stego_exp.retrieval import FeatureBank, embed_image


def build_bank_chunked(
    paths: list[str],
    out_npz: Path,
    n_clusters: int,
    chunk: int = 512,
) -> FeatureBank:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    feat_cache = out_npz.with_suffix(".feats.npy")
    path_cache = out_npz.with_suffix(".paths.json")

    if out_npz.exists():
        print(f"load bank {out_npz.name}")
        data = np.load(out_npz, allow_pickle=True)
        bank = FeatureBank()
        bank.paths = data["paths"].tolist()
        bank.feats = data["feats"]
        bank.clusters = {p: int(c) for p, c in zip(bank.paths, data["clusters"])}
        return bank

    # resume embeddings
    done_paths: list[str] = []
    feats_list: list[np.ndarray] = []
    if feat_cache.exists() and path_cache.exists():
        done_paths = json.loads(path_cache.read_text(encoding="utf-8"))
        arr = np.load(feat_cache)
        feats_list = [arr[i] for i in range(len(arr))]
        print(f"resume embeds {len(done_paths)}/{len(paths)}")

    done_set = set(done_paths)
    todo = [p for p in paths if p not in done_set]
    for i in tqdm(range(0, len(todo), chunk), desc="embed"):
        batch = todo[i : i + chunk]
        for p in batch:
            img = Image.open(p).convert("RGB")
            feats_list.append(embed_image(img))
            done_paths.append(p)
        # periodic save
        if (i // chunk) % 10 == 0 or i + chunk >= len(todo):
            np.save(feat_cache, np.stack(feats_list, axis=0))
            path_cache.write_text(json.dumps(done_paths), encoding="utf-8")
            print(f"  embed checkpoint {len(done_paths)}/{len(paths)}")

    feats = np.stack(feats_list, axis=0)
    k = min(n_clusters, max(2, len(paths) // 4))
    print(f"MiniBatchKMeans k={k} on {len(paths)} vectors...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3)
    labels = km.fit_predict(feats)

    bank = FeatureBank()
    bank.paths = list(done_paths)
    bank.feats = feats
    bank.clusters = {p: int(labels[i]) for i, p in enumerate(bank.paths)}
    bank.km = km

    np.savez_compressed(
        out_npz,
        paths=np.array(bank.paths, dtype=object),
        feats=bank.feats,
        clusters=np.array([bank.clusters[p] for p in bank.paths]),
    )
    print(f"saved {out_npz}")
    return bank


def bank_path_for(name: str, n_images: int) -> Path:
    return CACHE / f"bank_{name}_n{n_images}_k{N_CLUSTERS[name]}.npz"


if __name__ == "__main__":
    # smoke: build tiny bank
    from full_exp.prepare_detections import prepare_all

    dets = prepare_all()
    for name, cache in dets.items():
        preds = json.loads(cache.read_text(encoding="utf-8"))
        paths = sorted(p for p in preds if Path(p).exists())
        print(name, "usable", len(paths))
        build_bank_chunked(paths, bank_path_for(name, len(paths)), N_CLUSTERS[name])
