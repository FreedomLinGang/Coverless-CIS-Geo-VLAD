# -*- coding: utf-8 -*-
"""VLAD coarse retrieval + SIFT-J refine (no cluster/class auxiliary ID)."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from full_exp_cluster_sift.build_clusters import match_sift_in_cluster
from full_exp_sift.retrieval_sift import SIFTCache
from no_aux_hash.hashes import topk_indices
from no_aux_hash.he_index import vlad_vector
from no_aux_hash.rich_features import detect_sift


def _gallery_scores(g: np.ndarray, q: np.ndarray, f32_gallery: list) -> np.ndarray:
    """f32_gallery: [promoted_or_None, use_chunked_bool] as mutable holder."""
    q = q / (np.linalg.norm(q) + 1e-8)
    q32 = np.asarray(q, dtype=np.float32)
    if g.dtype != np.float16:
        return g @ q32
    promoted, use_chunked = f32_gallery
    if promoted is None and not use_chunked:
        try:
            f32_gallery[0] = np.asarray(g, dtype=np.float32)
            return f32_gallery[0] @ q32
        except MemoryError:
            f32_gallery[1] = True
    if f32_gallery[0] is not None:
        return f32_gallery[0] @ q32
    n = g.shape[0]
    out = np.empty(n, dtype=np.float32)
    bs = 8192
    for i in range(0, n, bs):
        out[i : i + bs] = np.asarray(g[i : i + bs], dtype=np.float32) @ q32
    return out


class VladRetriever:
    """SIFT VLAD-64 cosine shortlist, then Lowe+RANSAC maximize J."""

    def __init__(
        self,
        paths: list[str],
        gallery: np.ndarray,
        centers: np.ndarray,
        shortlist: int = 40,
        sift_cache_size: int = 16384,
    ):
        self.paths = list(paths)
        self.gallery = gallery
        self.centers = np.asarray(centers, dtype=np.float32)
        self.shortlist = int(shortlist)
        self.path_index = {p: i for i, p in enumerate(self.paths)}
        self._f32_holder: list = [None, False]
        self._sift_cache = SIFTCache(max_items=sift_cache_size)

    def shortlist_paths(self, img: Image.Image) -> list[str]:
        _, desc = detect_sift(img)
        q = vlad_vector(desc, self.centers)
        idx = topk_indices(_gallery_scores(self.gallery, q, self._f32_holder), self.shortlist, True)
        return [self.paths[i] for i in idx]

    def match_path(
        self,
        img: Image.Image,
        path_to_img: dict[str, Image.Image] | Any,
        cands: list[str] | None = None,
    ) -> str:
        cands = cands or self.shortlist_paths(img)
        return match_sift_in_cluster(
            img,
            cands,
            path_to_img,
            self._sift_cache,
            index=None,
            shortlist=self.shortlist,
        )

    def recover_with_recall(
        self,
        system,
        attacked_images: list[Image.Image],
        aux,
        pad_zeros: int,
        path_to_img,
    ) -> tuple[bytes | None, float, float]:
        if len(attacked_images) != len(aux):
            return None, 0.0, 0.0
        bits = ""
        hits_top1 = 0
        hits_pool = 0
        for img, rec in zip(attacked_images, aux):
            cands = self.shortlist_paths(img)
            cand_set = set(cands)
            if rec.path in self.path_index and rec.path in cand_set:
                hits_pool += 1
            found = self.match_path(img, path_to_img, cands)
            if found == rec.path:
                hits_top1 += 1
            atoms = system._atoms_for(found) if found else []
            rank_to_obj = {rank: oi for oi, rank in enumerate(rec.order) if rank > 0}
            for rank in sorted(rank_to_obj):
                oi = rank_to_obj[rank]
                if oi >= len(atoms):
                    bits += "0" * system.segment_bits
                    continue
                code = system.codebook.get(atoms[oi], "0" * system.segment_bits)
                bits += code
        n = max(1, len(aux))
        if pad_zeros:
            bits = bits[: max(0, len(bits) - pad_zeros)]
        if len(bits) % 8:
            bits = bits[: len(bits) - (len(bits) % 8)]
        try:
            msg = int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""
        except Exception:
            msg = None
        return msg, hits_top1 / n, hits_pool / n
