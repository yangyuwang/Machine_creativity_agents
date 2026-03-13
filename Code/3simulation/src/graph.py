"""
Build style kNN graph (cosine). Output nodes.csv (id, path, S, M, I) and edges_knn.csv (u, v, sim).
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Remove NaN/Inf and L2-normalize. Return float64 for sklearn numerical stability."""
    x = np.asarray(embeddings, dtype=np.float64)
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return (x / norms).astype(np.float64)


def build_knn_cosine(
    embeddings: np.ndarray,
    k: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """kNN with cosine metric. Returns (indices [N, k], similarities [N, k]). L2-normalized => cosine_sim = dot."""
    emb = _sanitize_embeddings(embeddings)
    n = len(emb)
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine", algorithm="brute")
    nn.fit(emb)
    dist, idx = nn.kneighbors(emb)
    # dist is cosine distance = 1 - cosine_sim; drop self
    idx = idx[:, 1:]
    dist = dist[:, 1:]
    sim = 1.0 - dist.astype(np.float32)
    return idx, sim


def run(
    embeddings: np.ndarray,
    paths: List[str],
    output_dir: Path,
    k: int = 15,
) -> pd.DataFrame:
    """Build kNN graph; save nodes (id, path, S, M, I) and edges_knn (source, target, sim)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(embeddings)
    indices, sims = build_knn_cosine(embeddings, k=k)

    nodes = pd.DataFrame({
        "id": np.arange(n, dtype=int),
        "path": paths,
        "S": 1.0,
        "M": 1.0,
        "I": 1.0,
    })
    nodes.to_csv(output_dir / "nodes.csv", index=False)

    edge_set = {}
    for i in range(n):
        for jj, j in enumerate(indices[i]):
            j = int(j)
            u, v = min(i, j), max(i, j)
            if (u, v) in edge_set:
                continue
            edge_set[(u, v)] = float(sims[i, jj])
    edges = pd.DataFrame([
        {"source": u, "target": v, "sim": s} for (u, v), s in sorted(edge_set.items())
    ])
    edges.to_csv(output_dir / "edges_knn.csv", index=False)
    return nodes
