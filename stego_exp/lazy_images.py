"""Lazy image dict + feature bank that streams from disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .retrieval import FeatureBank, embed_image


class LazyImages:
    """path -> PIL Image, loaded on demand (optionally cached LRU-ish)."""

    def __init__(self, max_cache: int = 256):
        self._cache: dict[str, Image.Image] = {}
        self._order: list[str] = []
        self.max_cache = max_cache
        self.paths: list[str] = []

    def add_paths(self, paths: list[str] | list[Path]):
        self.paths = [str(p) for p in paths]

    def __contains__(self, path: str) -> bool:
        return path in set(self.paths) or path in self._cache

    def keys(self):
        return self.paths

    def __iter__(self):
        return iter(self.paths)

    def items(self):
        for p in self.paths:
            yield p, self[p]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, path: str) -> Image.Image:
        if path in self._cache:
            return self._cache[path]
        img = Image.open(path).convert("RGB")
        self._cache[path] = img
        self._order.append(path)
        while len(self._order) > self.max_cache:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return img

    def get(self, path: str, default=None):
        try:
            return self[path]
        except Exception:
            return default


def build_bank_from_paths(paths: list[str], n_clusters: int = 64) -> FeatureBank:
    bank = FeatureBank(device="cpu")
    bank.paths = list(paths)
    feats = []
    for p in tqdm(paths, desc="embed"):
        img = Image.open(p).convert("RGB")
        feats.append(embed_image(img))
    bank.feats = np.stack(feats, axis=0)
    from sklearn.cluster import MiniBatchKMeans

    k = min(n_clusters, max(2, len(paths) // 4))
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=512, n_init=3)
    labels = km.fit_predict(bank.feats)
    bank.clusters = {p: int(labels[i]) for i, p in enumerate(bank.paths)}
    bank.km = km
    return bank
