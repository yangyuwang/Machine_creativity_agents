"""
UMAP 2D, clustering (visible-set k-means per round), modularity, classicization (top10% share, Gini).
"""
from pathlib import Path
from typing import Tuple

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

try:
    import umap
except ImportError:
    umap = None


def top10_share(in_deg: np.ndarray) -> float:
    """Top 10% in-degree share: sum(top-k indeg) / sum(all indeg)."""
    in_deg = np.asarray(in_deg, dtype=np.int64)
    total = float(in_deg.sum())  # equals total citation edges in directed graph
    if total <= 0:
        return np.nan
    n = len(in_deg)
    k = max(1, int(0.1 * n))
    top_sum = float(np.sort(in_deg)[-k:].sum())
    return top_sum / total


def gini_nonneg(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0:
        return np.nan
    s = x.sum()
    if s == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2 * np.dot(np.arange(1, n + 1), x) / (n * s)) - (n + 1) / n)


def _sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Remove NaN/Inf and L2-normalize. float64 for sklearn/UMAP numerical stability."""
    x = np.asarray(embeddings, dtype=np.float64)
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return (x / norms).astype(np.float64)


def pca_2d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """2D projection via PCA. Stable, no C/numba; avoids UMAP segfault on some stacks."""
    x = _sanitize_embeddings(embeddings)
    pca = PCA(n_components=min(2, x.shape[0], x.shape[1]), random_state=seed)
    out = pca.fit_transform(x)
    if out.shape[1] < 2:
        out = np.column_stack([out, np.zeros(len(out))])
    return out.astype(np.float64)


def umap_2d(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42) -> np.ndarray:
    if umap is None:
        raise ImportError("Install umap-learn")
    x = _sanitize_embeddings(embeddings)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed, metric="cosine", n_jobs=1)
    return reducer.fit_transform(x).astype(np.float64)


def cluster_stats(
    embeddings: np.ndarray,
    k: int = 8,
    seed: int = 42,
) -> Tuple[int, float, float, np.ndarray]:
    """Fixed k-means on full set (for UMAP labels). Returns (n_clusters, intra_avg, inter_avg, labels)."""
    n = len(embeddings)
    k = min(max(2, k), n - 1) if n > 1 else 1
    if k < 2:
        return 1, 0.0, 0.0, np.zeros(n, dtype=int)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    intra_dists = []
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        pts = embeddings[mask]
        intra_dists.extend(np.linalg.norm(pts - centers[c], axis=1).tolist())
    intra_avg = float(np.mean(intra_dists)) if intra_dists else 0.0
    if k >= 2:
        inter_d = pairwise_distances(centers)
        np.fill_diagonal(inter_d, np.nan)
        inter_avg = float(np.nanmean(inter_d))
    else:
        inter_avg = 0.0
    return k, intra_avg, inter_avg, labels


def cluster_stats_visible(
    embeddings: np.ndarray,
    visible_ids: np.ndarray,
    in_deg: np.ndarray,
    top_k: int = 80,
    k_means_k: int = 8,
    seed: int = 42,
) -> Tuple[int, float, float]:
    """Cluster the visible set (cumulative selected), optionally top-K by in_degree."""
    n = len(embeddings)
    if visible_ids.size == 0:
        return 0, np.nan, np.nan
    vis = np.unique(np.asarray(visible_ids, dtype=int))
    vis = vis[(vis >= 0) & (vis < n)]
    if vis.size == 0:
        return 0, np.nan, np.nan
    if vis.size > top_k:
        vis_sorted = sorted(vis.tolist(), key=lambda idx: (-float(in_deg[idx]), int(idx)))
        vis = np.asarray(vis_sorted[:top_k], dtype=int)
    sub = embeddings[vis]
    k = min(k_means_k, max(2, len(sub) // 5), len(sub) - 1) if len(sub) >= 2 else 1
    if k < 2:
        return 1, 0.0, 0.0
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(sub)
    centers = kmeans.cluster_centers_
    intra_dists = []
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        pts = sub[mask]
        intra_dists.extend(np.linalg.norm(pts - centers[c], axis=1).tolist())
    intra_avg = float(np.mean(intra_dists)) if intra_dists else 0.0
    if k >= 2:
        inter_d = pairwise_distances(centers)
        np.fill_diagonal(inter_d, np.nan)
        inter_avg = float(np.nanmean(inter_d))
    else:
        inter_avg = 0.0
    return k, intra_avg, inter_avg


def _in_degree_up_to(edges_df: pd.DataFrame, n_nodes: int, round_max: int) -> np.ndarray:
    """In-degree of each node from citation edges with round <= round_max."""
    df = edges_df[edges_df["round"] <= round_max]
    if df.empty:
        return np.zeros(n_nodes, dtype=np.float64)
    df = df.astype({"source": int, "target": int})
    G = nx.from_pandas_edgelist(df, "source", "target", create_using=nx.DiGraph)
    node_list = list(range(n_nodes))
    in_deg_dict = dict(G.in_degree(node_list))
    return np.array([in_deg_dict.get(i, 0) for i in range(n_nodes)], dtype=np.float64)


def modularity_and_classicization(
    edges_df: pd.DataFrame,
    n_nodes: int,
    round_max: int = None,
) -> Tuple[float, float, float]:
    """Modularity (undirected), top10% share, Gini. Returns (mod, top10, gini); no edges -> (0, nan, nan)."""
    if round_max is not None:
        edges_df = edges_df[edges_df["round"] <= round_max]
    if edges_df.empty:
        return 0.0, np.nan, np.nan
    edges_df = edges_df.astype({"source": int, "target": int})
    G = nx.from_pandas_edgelist(edges_df, "source", "target", create_using=nx.DiGraph)
    node_list = list(range(n_nodes))
    in_deg_dict = dict(G.in_degree(node_list))
    in_deg = np.array([in_deg_dict.get(i, 0) for i in range(n_nodes)], dtype=np.float64)
    try:
        Gu = G.to_undirected()
        comp = nx.community.greedy_modularity_communities(Gu)
        mod = float(nx.community.modularity(Gu, comp))
    except Exception:
        mod = 0.0
    total_in = in_deg.sum()
    if total_in <= 0:
        return mod, np.nan, np.nan
    top10 = top10_share(in_deg)
    gini = gini_nonneg(in_deg)
    return mod, top10, gini


def _partition_from_labels(labels: np.ndarray, n_nodes: int) -> list[set[int]]:
    """Build partition for networkx modularity from label vector over all nodes."""
    labels = np.asarray(labels, dtype=int)
    groups: dict[int, set[int]] = {}
    for i in range(min(n_nodes, len(labels))):
        lab = int(labels[i])
        groups.setdefault(lab, set()).add(i)
    # Keep only non-empty sets to satisfy nx modularity input requirements.
    return [g for g in groups.values() if g]


def _safe_modularity(Gu: nx.Graph, partition: list[set[int]]) -> float:
    if Gu.number_of_edges() == 0 or not partition:
        return 0.0
    try:
        return float(nx.community.modularity(Gu, partition))
    except Exception:
        return 0.0


def run_all(
    embeddings: np.ndarray,
    edges_citation: pd.DataFrame,
    n_nodes: int,
    output_dir: Path,
    obs_rounds: list,
    top_k_visible: int = 80,
    k_means_k: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Per obs_round metrics with explicit citation-evolving and embedding-static semantics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_clean = _sanitize_embeddings(embeddings)
    # Use PCA by default to avoid UMAP/numba segfault on some envs; set RENAISSANCE_USE_UMAP=1 to use UMAP
    if os.environ.get("RENAISSANCE_USE_UMAP", "").strip() == "1" and umap is not None:
        coords_2d = umap_2d(emb_clean, seed=seed)
    else:
        coords_2d = pca_2d(emb_clean, seed=seed)
    _, _, _, labels_embedding = cluster_stats(emb_clean, k=k_means_k, seed=seed)
    clusters_all = int(np.unique(labels_embedding).size)
    emb_partition = _partition_from_labels(labels_embedding, n_nodes)

    rows = []
    embed_hashes = []
    citation_hashes = []
    prev_r = None
    prev_topk_ids: np.ndarray | None = None
    for r in obs_rounds:
        # Evolving citation graph up to round r (authoritative source for temporal network metrics)
        df_r = edges_citation[edges_citation["round"] <= r].copy()
        if not df_r.empty:
            df_r = df_r.astype({"source": int, "target": int})
            Gd = nx.from_pandas_edgelist(df_r, "source", "target", create_using=nx.DiGraph)
        else:
            Gd = nx.DiGraph()
        Gd.add_nodes_from(range(n_nodes))
        Gu = Gd.to_undirected()
        n_g, m_g = Gu.number_of_nodes(), Gu.number_of_edges()

        # Citation-only dynamic communities (Scheme A component)
        if m_g > 0:
            communities_citation = list(nx.community.greedy_modularity_communities(Gu))
            modularity_citation = _safe_modularity(Gu, communities_citation)
            n_clusters_citation = len(communities_citation)
            labels_citation = np.full(n_nodes, -1, dtype=int)
            for cid, comm in enumerate(communities_citation):
                for node in comm:
                    labels_citation[int(node)] = cid
        else:
            modularity_citation = 0.0
            n_clusters_citation = 0
            labels_citation = np.full(n_nodes, -1, dtype=int)

        # Embedding-static partition evaluated on evolving citation graph (Scheme B component)
        modularity_embedding_partition_on_citation = _safe_modularity(Gu, emb_partition)

        # Degree distribution diagnostics (all nodes)
        in_deg = np.array([Gd.in_degree(i) for i in range(n_nodes)], dtype=np.float64)
        top10_all = top10_share(in_deg)
        gini_all = gini_nonneg(in_deg)
        active = in_deg > 0
        top10_active = top10_share(in_deg[active]) if np.any(active) else np.nan
        gini_active = gini_nonneg(in_deg[active]) if np.any(active) else np.nan
        k_turn = min(max(1, int(top_k_visible)), n_nodes)
        cur_topk_ids = np.argsort(-in_deg)[:k_turn]
        if prev_topk_ids is None:
            topk_turnover_rate = np.nan
        else:
            overlap = len(set(cur_topk_ids.tolist()) & set(prev_topk_ids.tolist()))
            topk_turnover_rate = float(1.0 - (overlap / max(1, k_turn)))
        prev_topk_ids = cur_topk_ids.copy()

        # Visible-set embedding clustering (cumulative selected targets; reflects visible world)
        visible_ids = df_r["target"].to_numpy(dtype=int) if not df_r.empty else np.array([], dtype=int)
        n_clusters_visible, intra_avg, inter_avg = cluster_stats_visible(
            emb_clean,
            visible_ids=visible_ids,
            in_deg=in_deg,
            top_k=top_k_visible,
            k_means_k=k_means_k,
            seed=seed,
        )
        separation_ratio = float(inter_avg / intra_avg) if intra_avg > 1e-12 else np.nan

        # Cross-cluster citation rate on newly added edges in this observation window.
        if prev_r is None:
            df_new = edges_citation[edges_citation["round"] <= r].copy()
        else:
            df_new = edges_citation[(edges_citation["round"] > prev_r) & (edges_citation["round"] <= r)].copy()
        if df_new.empty:
            cross_cluster_citation_rate = np.nan
        else:
            src = df_new["source"].to_numpy(dtype=int)
            dst = df_new["target"].to_numpy(dtype=int)
            valid = (src >= 0) & (src < n_nodes) & (dst >= 0) & (dst < n_nodes)
            if np.any(valid):
                src_lab = labels_embedding[src[valid]]
                dst_lab = labels_embedding[dst[valid]]
                cross_cluster_citation_rate = float(np.mean(src_lab != dst_lab))
            else:
                cross_cluster_citation_rate = np.nan

        emb_hash = hash(labels_embedding.tobytes())
        cit_hash = hash(labels_citation.tobytes())
        embed_hashes.append(emb_hash)
        citation_hashes.append(cit_hash)

        # Round-wise self-check logs (stdout) for quick diagnosis
        p50, p90, p99 = np.percentile(in_deg, [50, 90, 99]).tolist()
        zero_ratio = float(np.mean(in_deg == 0))
        print(
            f"[metrics][round={r}] graph_source=edges_citation n={n_g} m={m_g} "
            f"labels_embedding_unique={np.unique(labels_embedding).size} labels_embedding_hash={emb_hash} "
            f"labels_citation_unique={np.unique(labels_citation[labels_citation>=0]).size} labels_citation_hash={cit_hash}"
        )
        print(
            f"[metrics][round={r}] in_degree_summary max={in_deg.max():.0f} p50={p50:.2f} "
            f"p90={p90:.2f} p99={p99:.2f} zero_ratio={zero_ratio:.3f}"
        )
        top10_primary = top10_active if np.isfinite(top10_active) else top10_all
        # Requested concise classicization debug line
        print(
            f"[metrics][classicization] round={r} total_edges={int(in_deg.sum())} "
            f"max_indeg={int(in_deg.max())} top10_share={top10_primary:.6f} "
            f"top10_share_all={top10_all:.6f} top10_share_active={top10_active:.6f} gini={gini_all:.6f}"
        )

        rows.append({
            "round": r,
            # Backward-compatible aliases now aligned to citation-evolving communities.
            "n_clusters": n_clusters_citation,
            "modularity": modularity_citation,
            "clusters_all": clusters_all,
            "clusters_visible": n_clusters_visible,
            "cross_cluster_citation_rate": cross_cluster_citation_rate,
            # Explicit semantic columns.
            "n_clusters_citation": n_clusters_citation,
            "n_clusters_embedding_visible": n_clusters_visible,
            "intra_cluster_mean_dist": intra_avg,
            "inter_cluster_centroid_dist": inter_avg,
            "separation_ratio": separation_ratio,
            "modularity_citation": modularity_citation,
            "modularity_embedding_partition_on_citation": modularity_embedding_partition_on_citation,
            "in_degree_top10_share": top10_primary,
            "in_degree_top10_share_all": top10_all,
            "gini_in_degree": gini_all,
            "in_degree_top10_share_active": top10_active,
            "gini_in_degree_active": gini_active,
            "topk_turnover_rate": topk_turnover_rate,
        })
        prev_r = r

    if len(set(embed_hashes)) == 1:
        print("[metrics][warning] embedding labels hash is constant across rounds (static embedding partition).")
    if len(set(citation_hashes)) == 1:
        print("[metrics][warning] citation community labels hash is constant across rounds.")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    np.save(output_dir / "umap_2d.npy", coords_2d)
    np.save(output_dir / "cluster_labels.npy", labels_embedding)
    return metrics_df
