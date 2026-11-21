# -*- coding: utf-8 -*-
"""Streaming VLAD-64 bank (one codebook per dataset)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from full_exp_vlad_sift.config import CACHE, PCA_DIM, PCA_MIN_IMAGES, VLAD_K
from no_aux_hash.he_index import vlad_vector
from no_aux_hash.rich_features import bow_fit, detect_sift

_WORKER_CENTERS = None
_WORKER_SIFT = None


def _init_encode_worker(centers: np.ndarray):
    global _WORKER_CENTERS, _WORKER_SIFT
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2

    cv2.setNumThreads(1)
    _WORKER_CENTERS = centers
    _WORKER_SIFT = cv2.SIFT_create()


def _encode_one_path(path: str) -> np.ndarray:
    import cv2
    from PIL import Image

    from full_exp_cluster_sift.config import SIFT_SIZE
    from no_aux_hash.he_index import vlad_vector

    img = Image.open(path).convert("RGB")
    gray = cv2.cvtColor(np.asarray(img.resize((SIFT_SIZE, SIFT_SIZE))), cv2.COLOR_RGB2GRAY)
    _, desc = _WORKER_SIFT.detectAndCompute(gray, None)
    if desc is not None:
        desc = desc.astype(np.float32)
    return vlad_vector(desc, _WORKER_CENTERS)


def _safe_unlink(p: Path):
    try:
        if p.exists():
            p.unlink()
    except OSError as e:
        print(f"  skip unlink {p.name}: {e}")


def vlad_path(name: str, n: int, k: int = VLAD_K):
    return CACHE / f"vlad{k}_{name}_n{n}.npz"


def load_or_build_vlad(name: str, paths: list[str], images, k: int = VLAD_K, sample_n: int = 4000):
    print(f"load_or_build_vlad {name} n={len(paths)}", flush=True)
    out = vlad_path(name, len(paths), k)
    if out.exists():
        data = np.load(out, allow_pickle=True)
        if data["paths"].tolist() == paths:
            print(f"load VLAD {out.name}")
            return data["vlad"].astype(np.float32), data["centers"].astype(np.float32)

    n = len(paths)
    dim = k * 128
    mmap_p = CACHE / f"vlad{k}_{name}_n{n}.mmap"
    centers_p = CACHE / f"vlad{k}_{name}_n{n}.centers.npz"
    meta_p = CACHE / f"vlad{k}_{name}_n{n}.meta.npz"
    done_p = CACHE / f"vlad{k}_{name}_n{n}.done.npy"
    mmap_ok = mmap_p.exists() and mmap_p.stat().st_size == n * dim * 4
    if mmap_ok:
        centers = None
        if centers_p.exists() and centers_p.stat().st_size > 256:
            with np.load(centers_p, allow_pickle=True) as cdata:
                if int(cdata["n"]) == n:
                    centers = cdata["centers"].astype(np.float32)
        if centers is None and meta_p.exists():
            with np.load(meta_p, allow_pickle=True) as meta:
                centers = meta["centers"].astype(np.float32)
            np.savez_compressed(centers_p, centers=centers, n=np.int32(n))
        if centers is not None:
            print(f"load VLAD memmap {mmap_p.name}")
            return np.memmap(mmap_p, dtype=np.float32, mode="r", shape=(n, dim)), centers

    alt = Path(__file__).resolve().parents[1] / "no_aux_hash" / "cache" / f"he32_vlad64_{name}_n{len(paths)}.npz"
    if alt.exists() and k == 64:
        data = np.load(alt, allow_pickle=True)
        if "paths" in data.files and data["paths"].tolist() == paths:
            print(f"import VLAD from {alt.name}")
            vlad = data["vlad"].astype(np.float32)
            centers = data["centers_vlad"].astype(np.float32)
            np.savez_compressed(out, paths=np.array(paths, dtype=object), vlad=vlad, centers=centers)
            return vlad, centers

    meta_p = CACHE / f"vlad{k}_{name}_n{n}.meta.npz"
    done_p = CACHE / f"vlad{k}_{name}_n{n}.done.npy"
    done = 0
    centers = None
    if meta_p.exists() and mmap_p.exists():
        with np.load(meta_p, allow_pickle=True) as meta:
            saved_paths = meta["paths"].tolist() if "paths" in meta.files else None
            meta_centers = meta["centers"].astype(np.float32)
            meta_done = int(meta["done"]) if "done" in meta.files else 0
        if saved_paths is None or saved_paths == paths:
            done = int(np.load(done_p)) if done_p.exists() else meta_done
            centers = meta_centers
            print(f"  resume VLAD mmap {done}/{n}")

    if centers is None:
        rng = np.random.default_rng(0)
        n_s = min(sample_n, n)
        sample_idx = rng.choice(n, size=n_s, replace=False)
        sampled = []
        for i in tqdm(sample_idx, desc=f"{name} VLAD-vocab"):
            _, desc = detect_sift(images[paths[int(i)]])
            if desc is not None and len(desc):
                sampled.append(desc)
        print(f"fit VLAD codebook k={k} from {len(sampled)} images")
        centers = bow_fit(sampled, k=k, seed=0)
        np.savez_compressed(meta_p, paths=np.array(paths, dtype=object), centers=centers)

    mode = "r+" if mmap_p.exists() else "w+"
    vlad_mm = np.memmap(mmap_p, dtype=np.float32, mode=mode, shape=(n, dim))
    n_workers = 8
    todo = paths[done:]
    if todo:
        print(f"VLAD-encode {len(todo)} images with {n_workers} workers (resume {done}/{n})")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_encode_worker,
            initargs=(centers,),
        ) as ex:
            for j, vec in enumerate(
                tqdm(ex.map(_encode_one_path, todo, chunksize=16), total=len(todo), desc=f"{name} VLAD-encode")
            ):
                i = done + j
                vlad_mm[i] = vec
                if (i + 1) % 512 == 0 or i + 1 == n:
                    vlad_mm.flush()
                    np.save(done_p, np.int32(i + 1))
    vlad_mm.flush()
    del vlad_mm
    if n >= 40000:
        np.savez_compressed(centers_p, centers=centers, n=np.int32(n))
        _safe_unlink(meta_p)
        _safe_unlink(done_p)
        print(f"kept memmap {mmap_p} ({n} x {dim})")
        return np.memmap(mmap_p, dtype=np.float32, mode="r", shape=(n, dim)), centers

    vlad = np.memmap(mmap_p, dtype=np.float32, mode="r", shape=(n, dim))
    vlad = np.array(vlad, dtype=np.float32, copy=True)
    np.savez_compressed(out, paths=np.array(paths, dtype=object), vlad=vlad, centers=centers)
    if mmap_p.exists():
        mmap_p.unlink()
    if meta_p.exists():
        meta_p.unlink()
    if done_p.exists():
        done_p.unlink()
    print(f"saved {out}")
    return vlad, centers


def build_vlad_retriever(
    name: str,
    paths: list[str],
    images,
    shortlist: int = 40,
    k: int | None = None,
):
    """Build a VladRetriever from dataset gallery paths (uses cached VLAD bank if present)."""
    from stego_exp.vlad_retrieval import VladRetriever

    if k is None:
        k = VLAD_K
    g_raw, centers = load_or_build_vlad(name, paths, images, k=k)
    g_vlad, _ = project_search_matrix(name, g_raw)
    del g_raw
    cache_n = 8192 if len(paths) >= 40000 else 16384
    return VladRetriever(paths, g_vlad, centers, shortlist=shortlist, sift_cache_size=cache_n)


_F32_GALLERY = None


def gallery_scores(g: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine scores. Float16 gallery is promoted once to float32 for BLAS GEMV."""
    global _F32_GALLERY
    q = q / (np.linalg.norm(q) + 1e-8)
    q32 = np.asarray(q, dtype=np.float32)
    if g.dtype == np.float16:
        if _F32_GALLERY is None:
            try:
                print(f"promote gallery {g.shape} float16 -> float32 for GEMV", flush=True)
                _F32_GALLERY = np.asarray(g, dtype=np.float32)
            except MemoryError:
                print("float32 promote failed, chunked GEMV", flush=True)
                _F32_GALLERY = False
        if _F32_GALLERY is not False:
            return _F32_GALLERY @ q32
        n = g.shape[0]
        out = np.empty(n, dtype=np.float32)
        bs = 8192
        for i in range(0, n, bs):
            out[i : i + bs] = np.asarray(g[i : i + bs], dtype=np.float32) @ q32
        return out
    return g @ q32


def project_search_matrix(name: str, vlad, dim: int = PCA_DIM):
    """VOC: float32 VLAD in RAM. COCO-scale: float16 copy (~2GB) to keep full-dim cosine."""
    n, d = int(vlad.shape[0]), int(vlad.shape[1])
    if n < PCA_MIN_IMAGES:
        return np.array(vlad, dtype=np.float32, copy=True), None
    f16p = CACHE / f"vlad{VLAD_K}_{name}_n{n}.f16.mmap"
    need = n * d * 2
    if f16p.exists() and f16p.stat().st_size == need:
        print(f"load float16 gallery {f16p.name} ({need / 1e9:.2f} GB)")
        mm = np.memmap(f16p, dtype=np.float16, mode="r", shape=(n, d))
        g = np.array(mm, dtype=np.float16, copy=True)
        del mm
        return g, None
    print(f"write float16 gallery {f16p.name} ({need / 1e9:.2f} GB)")
    mm = np.memmap(f16p, dtype=np.float16, mode="w+", shape=(n, d))
    for i in range(0, n, 2048):
        mm[i : i + 2048] = np.asarray(vlad[i : i + 2048], dtype=np.float16)
    mm.flush()
    try:
        g = np.array(mm, dtype=np.float16, copy=True)
        del mm
        return g, None
    except MemoryError:
        print("float16 RAM copy failed, keep memmap")
        return mm, None
