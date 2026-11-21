"""Coverless steganography pipeline (proposed method only)."""
from __future__ import annotations

import secrets
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .codebook import build_codebook, select_objects_proposed, semantic_atom
from .retrieval import FeatureBank, retrieve_fast_hybrid
from .vlad_retrieval import VladRetriever


@dataclass
class StegoRecord:
    path: str
    order: list[int]  # permutation indices into selected objects; 0 means unused sentinel
    used_atoms: list[str]
    cluster: int  # legacy field; VLAD recovery ignores it
    segments: list[str]


@dataclass
class HideResult:
    stego_paths: list[str]
    aux: list[StegoRecord]
    pad_zeros: int
    message_bits: str
    n_stego: int


class CoverlessSystem:
    def __init__(
        self,
        detections: dict[str, list[dict]],
        path_to_img: dict[str, Image.Image],
        bank: FeatureBank,
        segment_bits: int = 6,
        area_ratio_thresh: float = 0.02,
        max_objects: int = 12,
        use_attributes: bool = False,
        retrieve_topk: int = 20,
        vlad_retriever: VladRetriever | None = None,
        **_ignored: Any,
    ):
        self.detections = detections
        self.path_to_img = path_to_img
        self.bank = bank
        self.segment_bits = segment_bits
        self.area_ratio_thresh = area_ratio_thresh
        self.max_objects = max_objects
        self.use_attributes = use_attributes
        self.retrieve_topk = retrieve_topk
        self.vlad_retriever = vlad_retriever

        def loader(p):
            return path_to_img[p]

        self.codebook = build_codebook(
            detections, segment_bits, use_attributes, image_loader=loader if use_attributes else None
        )
        self.inv_codebook = self._invert_codebook()
        self._atom_cache: dict[str, list[str]] = {}
        for p in path_to_img:
            self._atom_cache[p] = self._compute_atoms(p)
        self.index = self._build_index()  # code -> list[(path, obj_local_idx, atom)]

    def _selected(self, path: str) -> list[dict]:
        objs = self.detections.get(path, [])
        return select_objects_proposed(objs, self.area_ratio_thresh, self.max_objects)

    def _compute_atoms(self, path: str) -> list[str]:
        img = self.path_to_img[path] if self.use_attributes else None
        return [semantic_atom(o, img, self.use_attributes) for o in self._selected(path)]

    def _atoms_for(self, path: str) -> list[str]:
        return self._atom_cache.get(path) or self._compute_atoms(path)

    def _invert_codebook(self) -> dict[str, list[str]]:
        inv = defaultdict(list)
        for atom, code in self.codebook.items():
            inv[code].append(atom)
        return inv

    def _build_index(self) -> dict[str, list[tuple[str, int, str]]]:
        index: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        for path in self.path_to_img:
            atoms = self._atoms_for(path)
            for i, atom in enumerate(atoms):
                code = self.codebook.get(atom)
                if code is None:
                    continue
                index[code].append((path, i, atom))
        return index

    def _bits_from_message(self, message: bytes) -> tuple[str, int]:
        bits = "".join(f"{b:08b}" for b in message)
        pad = (self.segment_bits - (len(bits) % self.segment_bits)) % self.segment_bits
        if pad:
            bits = bits + ("0" * pad)
        return bits, pad

    def hide(self, message: bytes) -> HideResult | None:
        bits, pad = self._bits_from_message(message)
        segs = [bits[i : i + self.segment_bits] for i in range(0, len(bits), self.segment_bits)]
        remaining = segs[:]
        used_paths = []
        aux: list[StegoRecord] = []

        path_atoms = {p: self._atoms_for(p) for p in self.path_to_img}
        path_codes = {
            p: [self.codebook.get(a) for a in atoms] for p, atoms in path_atoms.items()
        }

        # Cover a contiguous PREFIX of remaining segments each time
        while remaining:
            need = remaining[0]
            cand_paths = [path for path, _, _ in self.index.get(need, [])]
            if not cand_paths:
                return None

            best_path = None
            best_maps: list[tuple[int, int, str]] = []  # (seg_offset, obj_idx, atom)

            for path in set(cand_paths):
                atoms = path_atoms[path]
                codes = path_codes[path]
                maps = []
                taken = set()
                for offset, seg in enumerate(remaining):
                    found = False
                    for oi, code in enumerate(codes):
                        if oi in taken:
                            continue
                        if code == seg:
                            maps.append((offset, oi, atoms[oi]))
                            taken.add(oi)
                            found = True
                            break
                    if not found:
                        break  # stop at first gap to keep contiguous prefix
                if len(maps) > len(best_maps):
                    best_maps = maps
                    best_path = path

            if not best_path or not best_maps:
                return None

            n_obj = len(path_atoms[best_path])
            order = [0] * n_obj
            used_atoms = [""] * n_obj
            covered_segs = []
            for rank, (offset, oi, atom) in enumerate(best_maps, start=1):
                order[oi] = rank
                used_atoms[oi] = atom
                covered_segs.append(remaining[offset])

            remaining = remaining[len(best_maps) :]

            aux.append(
                StegoRecord(
                    path=best_path,
                    order=order,
                    used_atoms=used_atoms,
                    cluster=self.bank.cluster_of(best_path),
                    segments=covered_segs,
                )
            )
            used_paths.append(best_path)

        return HideResult(
            stego_paths=used_paths,
            aux=aux,
            pad_zeros=pad,
            message_bits=bits,
            n_stego=len(used_paths),
        )

    def recover_from_paths(
        self,
        attacked_images: list[Image.Image],
        aux: list[StegoRecord],
        pad_zeros: int,
    ) -> tuple[bytes | None, float]:
        """Returns (message, retrieval_hit_rate)."""
        msg, r1, _ = recover_with_recall(
            self, attacked_images, aux, pad_zeros, self.path_to_img
        )
        return msg, r1

    def recover_with_recall_full(
        self,
        attacked_images: list[Image.Image],
        aux: list[StegoRecord],
        pad_zeros: int,
    ) -> tuple[bytes | None, float, float]:
        return recover_with_recall(self, attacked_images, aux, pad_zeros, self.path_to_img)


def recover_with_recall(
    system: CoverlessSystem,
    attacked_images,
    aux,
    pad_zeros,
    path_to_img,
) -> tuple[bytes | None, float, float]:
    """Recover message; also return R@1 and candidate-pool hit (R@pool)."""
    if system.vlad_retriever is not None:
        return system.vlad_retriever.recover_with_recall(
            system, attacked_images, aux, pad_zeros, path_to_img
        )
    if len(attacked_images) != len(aux):
        return None, 0.0, 0.0
    bits = ""
    hits_top1 = 0
    hits_pool = 0
    for img, rec in zip(attacked_images, aux):
        recalled = system.bank.recall_by_feature(img, topk=system.retrieve_topk)
        pool_hit = rec.path in set(recalled)
        found = retrieve_fast_hybrid(
            img,
            system.bank,
            path_to_img,
            cluster_hint=rec.cluster,
            topk=system.retrieve_topk,
        )
        if found == rec.path:
            hits_top1 += 1
        if pool_hit:
            hits_pool += 1

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


def random_message(n_bytes: int) -> bytes:
    return secrets.token_bytes(n_bytes)


def segment_accuracy(gt_bits: str, pred_msg: bytes | None, segment_bits: int, pad_zeros: int) -> float:
    if pred_msg is None:
        return 0.0
    pred_bits = "".join(f"{b:08b}" for b in pred_msg)
    clean = gt_bits[: len(gt_bits) - pad_zeros] if pad_zeros else gt_bits
    L = min(len(clean), len(pred_bits))
    clean, pred_bits = clean[:L], pred_bits[:L]
    if L < segment_bits:
        return 1.0 if clean == pred_bits else 0.0
    n = L // segment_bits
    ok = 0
    for i in range(n):
        a = clean[i * segment_bits : (i + 1) * segment_bits]
        b = pred_bits[i * segment_bits : (i + 1) * segment_bits]
        if a == b:
            ok += 1
    return ok / max(1, n)
