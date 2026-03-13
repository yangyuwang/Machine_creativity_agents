"""
Evolution: Church/Guild (Normative) + Court/Civic (Competitive) patronage;
S, M, I updates; citation edges with type (exemplar, peer, print).
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .graph import build_knn_cosine


def _center_and_d_center(embeddings: np.ndarray) -> np.ndarray:
    """Mainstream center (L2-normed mean). d_center[i] = 1 - cosine(emb[i], c) = deviation from mainstream."""
    emb = np.asarray(embeddings, dtype=np.float64)
    emb = np.nan_to_num(emb, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    emb = emb / norms
    center = emb.mean(axis=0)
    cnorm = np.linalg.norm(center) + 1e-8
    center = center / cnorm
    # cosine similarity (float64 avoids matmul overflow/underflow warnings)
    cos_sim = emb @ center
    cos_sim = np.clip(np.nan_to_num(cos_sim, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    return (1.0 - cos_sim).astype(np.float32)


def _sample_by_influence(
    n: int,
    influence: np.ndarray,
    sample_size: int,
) -> List[int]:
    """Weighted sample by influence without replacement."""
    if n <= 0 or sample_size <= 0:
        return []
    csize = min(int(sample_size), n)
    weights = np.maximum(np.asarray(influence, dtype=np.float64), 1e-8)
    probs = weights / weights.sum()
    # Use numpy for stable weighted sampling without replacement.
    sampled = np.random.choice(np.arange(n), size=csize, replace=False, p=probs)
    return [int(x) for x in sampled.tolist()]


def _top_fraction_ids(candidates_ids: List[int], scores: np.ndarray, frac: float) -> List[int]:
    if not candidates_ids:
        return []
    frac_eff = float(np.clip(frac, 0.0, 1.0))
    k = max(1, int(round(len(candidates_ids) * frac_eff)))
    ranked = sorted(candidates_ids, key=lambda idx: float(scores[idx]), reverse=True)
    return ranked[: min(k, len(ranked))]


def sample_exposure_galleries(
    *,
    node_ids: np.ndarray,
    influence: np.ndarray,
    gallery_size: int,
    rng: random.Random,
    master_ids: set[int] | None = None,
    master_boost: float = 1.0,
) -> Dict[int, List[int]]:
    """
    Exposure sampler (Yanjing side): decide who sees what.
    Returns per-artist visible gallery node ids.
    """
    ids = np.asarray(node_ids, dtype=int)
    n = len(ids)
    if n == 0:
        return {}
    gsize = max(1, min(int(gallery_size), max(1, n - 1)))
    inf = np.maximum(np.asarray(influence, dtype=np.float64), 1e-8).copy()
    if master_ids:
        for mid in master_ids:
            if 0 <= int(mid) < len(inf):
                inf[int(mid)] *= max(1.0, float(master_boost))
    out: Dict[int, List[int]] = {}
    for nid in ids.tolist():
        pool = [x for x in ids.tolist() if int(x) != int(nid)]
        if not pool:
            out[int(nid)] = []
            continue
        weights = np.asarray([inf[int(x)] if 0 <= int(x) < len(inf) else 1e-8 for x in pool], dtype=np.float64)
        probs = weights / max(1e-8, float(weights.sum()))
        kk = min(gsize, len(pool))
        chosen = np.random.choice(np.asarray(pool, dtype=int), size=kk, replace=False, p=probs)
        out[int(nid)] = [int(x) for x in chosen.tolist()]
    return out


def run_round(
    n: int,
    embeddings: np.ndarray,
    knn_indices: np.ndarray,
    S: np.ndarray,
    M: np.ndarray,
    I: np.ndarray,
    d_center: np.ndarray,
    wS: float,
    wM: float,
    tau: float,
    M_cand: int,
    p_norm: float,
    p_comp: float,
    s_gain: float,
    m_gain: float,
    mode: str,
    rng: random.Random,
    p_global: float,
    external_feedback: Dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    One round. Returns (S_new, M_new, I_new, citations list of (src, dst, type)).
    mode: baseline_random | norm_only | dual_patronage
    """
    # Scores
    score_norm = -d_center + wS * S
    novelty = d_center.copy()
    score_comp = novelty + wM * M
    comp_eligible = score_norm >= tau

    # A) Unified candidate sampling by influence
    candidates_ids = _sample_by_influence(n=n, influence=I, sample_size=M_cand)
    if not candidates_ids:
        round_data = {
            "candidates_ids": [],
            "selected_ids": [],
            "selected_norm_ids": [],
            "selected_comp_ids": [],
            "citations_added": [],
            "citation_records": [],
        }
        return S, M, I, round_data

    selected_norm_ids: List[int] = []
    selected_comp_ids: List[int] = []
    if mode == "baseline_random":
        keep = max(1, int(round(len(candidates_ids) * p_norm)))
        keep = min(keep, len(candidates_ids))
        selected_ids = rng.sample(candidates_ids, keep)
    else:
        selected_norm_ids = _top_fraction_ids(candidates_ids, score_norm, p_norm)
        if mode == "norm_only":
            selected_ids = list(selected_norm_ids)
        else:
            eligible_ids = [idx for idx in candidates_ids if bool(comp_eligible[idx])]
            if eligible_ids:
                selected_comp_ids = _top_fraction_ids(eligible_ids, score_comp, p_comp)
            selected_ids = list(dict.fromkeys(selected_norm_ids + selected_comp_ids))

    # C) Update capital
    S_new = S.copy()
    M_new = M.copy()
    I_new = I.copy()
    for i in selected_ids:
        I_new[i] += 1.0
    for i in selected_norm_ids:
        S_new[i] += s_gain
    for i in selected_comp_ids:
        M_new[i] += m_gain

    # E) Optional external closed-loop feedback (from generation/decode/citation stage).
    if external_feedback:
        fb_i = np.asarray(external_feedback.get("I_bonus", np.zeros(n)), dtype=np.float64)
        fb_s = np.asarray(external_feedback.get("S_bonus", np.zeros(n)), dtype=np.float64)
        fb_m = np.asarray(external_feedback.get("M_bonus", np.zeros(n)), dtype=np.float64)
        if len(fb_i) < n:
            fb_i = np.pad(fb_i, (0, n - len(fb_i)), mode="constant")
        if len(fb_s) < n:
            fb_s = np.pad(fb_s, (0, n - len(fb_s)), mode="constant")
        if len(fb_m) < n:
            fb_m = np.pad(fb_m, (0, n - len(fb_m)), mode="constant")
        I_new = I_new + np.nan_to_num(fb_i[:n], nan=0.0, posinf=0.0, neginf=0.0)
        S_new = S_new + np.nan_to_num(fb_s[:n], nan=0.0, posinf=0.0, neginf=0.0)
        M_new = M_new + np.nan_to_num(fb_m[:n], nan=0.0, posinf=0.0, neginf=0.0)
        I_new = np.clip(I_new, 0.1, None)
        S_new = np.clip(S_new, 0.1, None)
        M_new = np.clip(M_new, 0.1, None)

    # D) Citation edges: exemplar, peer, print
    citation_records: List[Tuple[int, int, str]] = []
    citations_added: List[Tuple[int, int]] = []
    for node in selected_ids:
        neighs = list(knn_indices[node])
        if not neighs:
            exemplar = node
            peer = node
        else:
            inf_neigh = [(I_new[j], j) for j in neighs]
            exemplar = max(inf_neigh, key=lambda x: x[0])[1]
            peer = rng.choice(neighs)
        citation_records.append((int(exemplar), int(node), "exemplar"))
        citation_records.append((int(peer), int(node), "peer"))
        citations_added.append((int(exemplar), int(node)))
        citations_added.append((int(peer), int(node)))
        if p_global > 0 and rng.random() < p_global:
            if n > 1:
                probs_glob = np.maximum(I_new, 1e-6)
                probs_glob = probs_glob / probs_glob.sum()
                print_ref = rng.choices(range(n), weights=probs_glob, k=1)[0]
            else:
                print_ref = node
            citation_records.append((int(print_ref), int(node), "print"))
            citations_added.append((int(print_ref), int(node)))
    round_data = {
        "candidates_ids": [int(x) for x in candidates_ids],
        "selected_ids": [int(x) for x in selected_ids],
        "selected_norm_ids": [int(x) for x in selected_norm_ids],
        "selected_comp_ids": [int(x) for x in selected_comp_ids],
        "citations_added": citations_added,
        "citation_records": citation_records,
    }
    return S_new, M_new, I_new, round_data


def run_evolution(
    embeddings: np.ndarray,
    paths: List[str],
    output_dir: Path,
    k: int = 15,
    T: int = 30,
    M_cand: int = 40,
    p_norm: float = 0.2,
    p_comp: float = 0.2,
    tau: float = 0.0,
    wS: float = 0.3,
    wM: float = 0.3,
    s_gain: float = 0.2,
    m_gain: float = 0.2,
    mode: str = "dual_patronage",
    seed: int = 42,
    game_top_k: int = 80,
    aS: float = 1.0,
    aM: float = 1.0,
    aR: float = 1.0,
    p_global_random: float = 0.1,
    p_global_norm: float = 0.02,
    p_global_dual: float = 0.2,
    init_i_bonus: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run T rounds; save nodes.csv, edges_citation.csv (source, target, round, type)."""
    rng = random.Random(seed)
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(embeddings)
    indices, _ = build_knn_cosine(embeddings, k=k)
    d_center = _center_and_d_center(embeddings)
    q20 = float(np.quantile(d_center, 0.2))
    q80 = float(np.quantile(d_center, 0.8))
    conform_mask = d_center <= q20
    diff_mask = d_center >= q80
    mid_mask = ~(conform_mask | diff_mask)

    S = np.ones(n, dtype=np.float64)
    M = np.ones(n, dtype=np.float64)
    I = np.ones(n, dtype=np.float64)
    if init_i_bonus is not None and len(init_i_bonus) == n:
        bonus = np.asarray(init_i_bonus, dtype=np.float64)
        bonus = np.nan_to_num(bonus, nan=0.0, posinf=0.0, neginf=0.0)
        I = I + np.clip(bonus, 0.0, None)
    payoff_cum = np.zeros(n, dtype=np.float64)

    nodes = pd.DataFrame({"id": np.arange(n), "path": paths, "S": S, "M": M, "I": I})
    all_citations = []
    game_rows = []

    for t in range(T):
        if mode == "baseline_random":
            p_global = float(p_global_random)
        elif mode == "norm_only":
            p_global = float(p_global_norm)
        else:
            p_global = float(p_global_dual)

        S, M, I, round_data = run_round(
            n, embeddings, indices, S, M, I, d_center,
            wS, wM, tau, M_cand, p_norm, p_comp, s_gain, m_gain, mode, rng,
            p_global=p_global,
        )
        selected_ids = round_data["selected_ids"]
        selected_norm_ids = round_data["selected_norm_ids"]
        selected_comp_ids = round_data["selected_comp_ids"]
        citation_records = round_data["citation_records"]
        for src, dst, typ in citation_records:
            all_citations.append({"source": src, "target": dst, "round": t, "type": typ})
        nodes["S"], nodes["M"], nodes["I"] = S, M, I

        # Evolutionary-game accounting
        payoff = np.zeros(n, dtype=np.float64)
        if mode == "baseline_random":
            if selected_ids:
                payoff[np.asarray(selected_ids, dtype=int)] += float(aR)
        else:
            if selected_norm_ids:
                payoff[np.asarray(selected_norm_ids, dtype=int)] += float(aS)
            if selected_comp_ids:
                payoff[np.asarray(selected_comp_ids, dtype=int)] += float(aM)
        payoff_cum += payoff
        k_eff = min(max(1, game_top_k), n)
        top_idx = np.argsort(-I)[:k_eff]
        share_conform_topK = float(np.mean(conform_mask[top_idx])) if k_eff > 0 else np.nan
        share_diff_topK = 1.0 - share_conform_topK if np.isfinite(share_conform_topK) else np.nan
        payoff_conform_mean = float(payoff[conform_mask].mean()) if conform_mask.any() else np.nan
        payoff_diff_mean = float(payoff[diff_mask].mean()) if diff_mask.any() else np.nan
        payoff_mid_mean = float(payoff[mid_mask].mean()) if mid_mask.any() else np.nan
        payoff_conform_cum_mean = float(payoff_cum[conform_mask].mean()) if conform_mask.any() else np.nan
        payoff_diff_cum_mean = float(payoff_cum[diff_mask].mean()) if diff_mask.any() else np.nan
        payoff_mid_cum_mean = float(payoff_cum[mid_mask].mean()) if mid_mask.any() else np.nan
        if (t % 5) == 0:
            n_conform = int(np.sum(conform_mask))
            n_diff = int(np.sum(diff_mask))
            n_mid = int(np.sum(mid_mask))
            print(
                f"[game-debug] mode={mode} round={t} "
                f"n_selected={len(selected_ids)} payoff_pos={int(np.sum(payoff > 0))} "
                f"sum_u_conform={float(payoff[conform_mask].sum()):.3f} "
                f"sum_u_diff={float(payoff[diff_mask].sum()):.3f} "
                f"n_conform={n_conform} n_diff={n_diff} n_mid={n_mid}"
            )
        game_rows.append({
            "round": t,
            "share_conform_topK": share_conform_topK,
            "share_diff_topK": share_diff_topK,
            "payoff_conform_mean": payoff_conform_mean,
            "payoff_diff_mean": payoff_diff_mean,
            "payoff_mid_mean": payoff_mid_mean,
            "payoff_conform_cum_mean": payoff_conform_cum_mean,
            "payoff_diff_cum_mean": payoff_diff_cum_mean,
            "payoff_mid_cum_mean": payoff_mid_cum_mean,
            "n_selected": int(len(selected_ids)),
            "n_selected_norm": int(len(selected_norm_ids)),
            "n_selected_comp": int(len(selected_comp_ids)),
        })

    # Snapshot after the final round as round=T (for obs_rounds that include T).
    if game_rows and game_rows[-1]["round"] != T:
        final_row = dict(game_rows[-1])
        final_row["round"] = T
        game_rows.append(final_row)

    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges_citation = pd.DataFrame(all_citations)
    edges_citation.to_csv(output_dir / "edges_citation.csv", index=False)
    game_df = pd.DataFrame(game_rows)
    game_df.to_csv(output_dir / "game_rounds.csv", index=False)
    return nodes, edges_citation, game_df  # One run_round per t; S,M,I and citations updated each round


def run_evolution_stateful(
    embeddings: np.ndarray,
    paths: List[str],
    round_start: int,
    round_count: int,
    mode: str,
    seed: int,
    k: int,
    M_cand: int,
    p_norm: float,
    p_comp: float,
    tau: float,
    wS: float,
    wM: float,
    s_gain: float,
    m_gain: float,
    game_top_k: int,
    aS: float,
    aM: float,
    aR: float,
    p_global_random: float,
    p_global_norm: float,
    p_global_dual: float,
    state: Dict[str, object] | None = None,
    external_feedback_by_round: Dict[int, Dict[str, np.ndarray]] | None = None,
) -> Dict[str, object]:
    """Stateful evolution segment for closed-loop spawn runs."""
    rng = random.Random(seed + int(round_start))
    np.random.seed(seed + int(round_start))
    n = len(embeddings)
    indices, _ = build_knn_cosine(embeddings, k=k)
    d_center = _center_and_d_center(embeddings)
    q20 = float(np.quantile(d_center, 0.2))
    q80 = float(np.quantile(d_center, 0.8))
    conform_mask = d_center <= q20
    diff_mask = d_center >= q80
    mid_mask = ~(conform_mask | diff_mask)

    if state is None:
        S = np.ones(n, dtype=np.float64)
        M = np.ones(n, dtype=np.float64)
        I = np.ones(n, dtype=np.float64)
        payoff_cum = np.zeros(n, dtype=np.float64)
        all_citations: list[dict] = []
        game_rows: list[dict] = []
    else:
        S = np.asarray(state["S"], dtype=np.float64)
        M = np.asarray(state["M"], dtype=np.float64)
        I = np.asarray(state["I"], dtype=np.float64)
        payoff_cum = np.asarray(state["payoff_cum"], dtype=np.float64)
        all_citations = list(state.get("all_citations", []))
        game_rows = list(state.get("game_rows", []))
        if len(S) < n:
            add = n - len(S)
            S = np.concatenate([S, np.ones(add, dtype=np.float64)])
            M = np.concatenate([M, np.ones(add, dtype=np.float64)])
            I = np.concatenate([I, np.ones(add, dtype=np.float64)])
            payoff_cum = np.concatenate([payoff_cum, np.zeros(add, dtype=np.float64)])

    for local_t in range(round_count):
        t = int(round_start + local_t)
        if mode == "baseline_random":
            p_global = float(p_global_random)
        elif mode == "norm_only":
            p_global = float(p_global_norm)
        else:
            p_global = float(p_global_dual)
        S, M, I, round_data = run_round(
            n, embeddings, indices, S, M, I, d_center,
            wS, wM, tau, M_cand, p_norm, p_comp, s_gain, m_gain, mode, rng,
            p_global=p_global,
            external_feedback=(external_feedback_by_round or {}).get(int(t)),
        )
        selected_ids = round_data["selected_ids"]
        selected_norm_ids = round_data["selected_norm_ids"]
        selected_comp_ids = round_data["selected_comp_ids"]
        citation_records = round_data["citation_records"]
        for src, dst, typ in citation_records:
            all_citations.append({"source": src, "target": dst, "round": t, "type": typ})

        payoff = np.zeros(n, dtype=np.float64)
        if mode == "baseline_random":
            if selected_ids:
                payoff[np.asarray(selected_ids, dtype=int)] += float(aR)
        else:
            if selected_norm_ids:
                payoff[np.asarray(selected_norm_ids, dtype=int)] += float(aS)
            if selected_comp_ids:
                payoff[np.asarray(selected_comp_ids, dtype=int)] += float(aM)
        payoff_cum += payoff
        k_eff = min(max(1, game_top_k), n)
        top_idx = np.argsort(-I)[:k_eff]
        share_conform_topK = float(np.mean(conform_mask[top_idx])) if k_eff > 0 else np.nan
        share_diff_topK = 1.0 - share_conform_topK if np.isfinite(share_conform_topK) else np.nan
        payoff_conform_mean = float(payoff[conform_mask].mean()) if conform_mask.any() else np.nan
        payoff_diff_mean = float(payoff[diff_mask].mean()) if diff_mask.any() else np.nan
        payoff_mid_mean = float(payoff[mid_mask].mean()) if mid_mask.any() else np.nan
        payoff_conform_cum_mean = float(payoff_cum[conform_mask].mean()) if conform_mask.any() else np.nan
        payoff_diff_cum_mean = float(payoff_cum[diff_mask].mean()) if diff_mask.any() else np.nan
        payoff_mid_cum_mean = float(payoff_cum[mid_mask].mean()) if mid_mask.any() else np.nan
        game_rows.append({
            "round": t,
            "share_conform_topK": share_conform_topK,
            "share_diff_topK": share_diff_topK,
            "payoff_conform_mean": payoff_conform_mean,
            "payoff_diff_mean": payoff_diff_mean,
            "payoff_mid_mean": payoff_mid_mean,
            "payoff_conform_cum_mean": payoff_conform_cum_mean,
            "payoff_diff_cum_mean": payoff_diff_cum_mean,
            "payoff_mid_cum_mean": payoff_mid_cum_mean,
            "n_selected": int(len(selected_ids)),
            "n_selected_norm": int(len(selected_norm_ids)),
            "n_selected_comp": int(len(selected_comp_ids)),
        })

    # Keep game rounds aligned with observation rounds that include segment end.
    final_round = int(round_start + max(0, round_count))
    if game_rows and int(game_rows[-1]["round"]) != final_round:
        final_row = dict(game_rows[-1])
        final_row["round"] = final_round
        game_rows.append(final_row)

    return {
        "S": S,
        "M": M,
        "I": I,
        "payoff_cum": payoff_cum,
        "all_citations": all_citations,
        "game_rows": game_rows,
        "paths": list(paths),
        "n": n,
    }
