# -*- coding: utf-8 -*-
"""Cluster index used as FeatureBank stand-in for CoverlessSystem."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class ClusterIndex:
    def __init__(self, paths: list[str], clusters: dict[str, int], feats: np.ndarray | None = None):
        self.paths = list(paths)
        self.clusters = dict(clusters)
        self.feats = feats
        self._by_c: dict[int, list[str]] = defaultdict(list)
        for p, c in self.clusters.items():
            self._by_c[int(c)].append(p)
        self.index = {p: i for i, p in enumerate(self.paths)}

    def cluster_of(self, path: str) -> int:
        return int(self.clusters[path])

    def candidates_by_cluster(self, cluster_id: int) -> list[str]:
        return self._by_c.get(int(cluster_id), [])
