#!/usr/bin/env python3
"""
One-shot run: embed -> graph -> evolve -> metrics -> viz.
Modes: baseline_random | norm_only | dual_patronage
Usage: python main.py [--mode dual_patronage] [--config config.json]
"""
import argparse
import faulthandler
import json
import os
import random
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Limit native thread pools early (before importing numpy/torch/sklearn).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# Suppress matmul overflow/divide-by-zero from sklearn/float32 on some stacks
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in matmul", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in matmul", category=RuntimeWarning)

from src.embed import extract_embeddings
from src.embed import run as run_embed
from src.batch_viz import render_batch_compare_plots
from src.generate import generate_round_images, run_generation
from src.graph import run as run_graph
from src.evolve import run_evolution, run_evolution_stateful, sample_exposure_galleries
from src.metrics import run_all as run_metrics
from src.vlm_eval import run_vlm_eval


def load_dotenv_file(dotenv_path: Path) -> None:
    """Minimal .env loader to avoid extra dependency."""
    if not dotenv_path.is_file():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("-")
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-_").lower()


def _fmt_num(x: float) -> str:
    s = f"{float(x):.2f}"
    return s.replace(".", "p")


def make_single_run_dir(
    *,
    project_dir: Path,
    pipeline: str,
    mode: str,
    strategy_policy: str,
    T: int,
    spawn_round_every: int,
    spawn_per_round: int,
    gen_top_k: int,
) -> Path:
    ts = now_compact()
    if pipeline == "env_only":
        tag = f"{ts}-{_slug(mode)}-t{int(T)}"
        return project_dir / "runs_env_only" / "single_runs" / tag
    tag = f"{ts}-{_slug(mode)}-{_slug(strategy_policy)}-t{int(T)}-sp{int(spawn_round_every)}x{int(spawn_per_round)}-gk{int(gen_top_k)}"
    return project_dir / "runs_env_plus_artist" / "single_runs" / tag


def make_batch_root_dir(
    *,
    project_dir: Path,
    output_root: str | None,
    batch_kind: str,
    T: int,
    spawn_round_every: int,
    spawn_per_round: int,
    gen_top_k: int,
    include_mixed_strategy: bool,
    config_path: Path,
) -> Path:
    if output_root:
        return (project_dir / output_root).resolve()
    grid = "3x4" if include_mixed_strategy else "3x3"
    cfg = _slug(config_path.stem or "config")
    if batch_kind == "env_only":
        tag = f"batch-3x1-{now_compact()}-{cfg}-t{int(T)}"
        return (project_dir / "runs_env_only" / "batch_runs" / tag).resolve()
    tag = f"batch-{grid}-{now_compact()}-{cfg}-t{int(T)}-sp{int(spawn_round_every)}x{int(spawn_per_round)}-gk{int(gen_top_k)}"
    return (project_dir / "runs_env_plus_artist" / "batch_runs" / tag).resolve()


def build_batch_run_specs(
    *,
    batch_kind: str = "env_plus_artist",
    include_mixed_strategy: bool = False,
    repeats: int = 1,
    batch_ts: str | None = None,
) -> list[dict]:
    """
    Build batch run specs.
    - env_only: 3x1 (three environments; strategies disabled), optional repeats
    - env_plus_artist: 3x3 (env x strategy), optional mixed row and repeats
    """
    env_modes = [
        ("E1", "baseline_random"),
        ("E2", "norm_only"),
        ("E3", "dual_patronage"),
    ]
    strategies = [
        ("S1", "imitation_only", 0.0),
        ("S2", "differentiation_only", 1.0),
        ("S3", "self_consistency_only", 0.0),
    ]
    if include_mixed_strategy and batch_kind != "env_only":
        strategies.insert(0, ("S0", "mixed", 0.3))
    runs = []
    rep_n = max(1, int(repeats))
    if batch_kind == "env_only":
        for env_idx, (_e_tag, mode) in enumerate(env_modes, start=1):
            for rep in range(1, rep_n + 1):
                runs.append(
                    {
                        "run_id": f"e{env_idx}-r{rep}-{_slug(mode)}",
                        "mode": mode,
                        "strategy_policy": "none",
                        "rebel_ratio": 0.0,
                    }
                )
        return runs
    for env_idx, (_e_tag, mode) in enumerate(env_modes, start=1):
        for strat_idx, (_s_tag, strategy_policy, rebel_ratio) in enumerate(strategies, start=1):
            for rep in range(1, rep_n + 1):
                runs.append(
                    {
                        "run_id": (
                            f"e{env_idx}-s{strat_idx}-r{rep}-"
                            f"{_slug(mode)}-{_slug(strategy_policy)}-rr{_fmt_num(rebel_ratio)}"
                        ),
                        "mode": mode,
                        "strategy_policy": strategy_policy,
                        "rebel_ratio": rebel_ratio,
                    }
                )
    return runs


def extract_batch_summary(metrics_path: Path, run_id: str) -> dict:
    if not metrics_path.is_file():
        return {"run_id": run_id, "status": "missing_metrics"}
    df = pd.read_csv(metrics_path)
    if df.empty:
        return {"run_id": run_id, "status": "empty_metrics"}
    row = df.sort_values("round").iloc[-1]
    out = {"run_id": run_id, "status": "ok", "round_final": int(row.get("round", -1))}
    keep_cols = [
        "modularity",
        "in_degree_top10_share",
        "gini_in_degree",
        "cross_cluster_citation_rate",
        "separation_ratio",
        "share_conform_topK",
        "share_diff_topK",
        "payoff_conform_cum_mean",
        "payoff_diff_cum_mean",
        "topk_turnover_rate",
        "generated_count",
    ]
    for c in keep_cols:
        out[c] = float(row[c]) if c in row and pd.notna(row[c]) else np.nan
    return out


def infer_artist_types(
    nodes_df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    master_ratio: float = 0.1,
    rebel_ratio: float = 0.3,
    strategy_policy: str = "mixed",
) -> dict[int, str]:
    """Infer artist strategy type for closed-loop adapt step."""
    n = len(nodes_df)
    if n == 0:
        return {}
    strategy_policy = str(strategy_policy or "mixed").strip().lower()
    if strategy_policy in {"imitation_only", "imitation"}:
        return {int(r["id"]): "follower" for _, r in nodes_df.iterrows()}
    if strategy_policy in {"differentiation_only", "differentiation"}:
        return {int(r["id"]): "rebel" for _, r in nodes_df.iterrows()}
    if strategy_policy in {"self_consistency_only", "self_consistency"}:
        return {int(r["id"]): "master" for _, r in nodes_df.iterrows()}
    em = np.asarray(embeddings, dtype=np.float64)
    em = np.nan_to_num(em, nan=0.0, posinf=1.0, neginf=-1.0)
    norms = np.linalg.norm(em, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    em = em / norms
    center = em.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-8)
    d_center = 1.0 - np.clip(em @ center, -1.0, 1.0)
    top_n = max(1, int(round(n * float(np.clip(master_ratio, 0.01, 0.5)))))
    master_ids = set(nodes_df.sort_values("I", ascending=False).head(top_n)["id"].astype(int).tolist())
    remain = [int(nodes_df.iloc[i]["id"]) for i in range(n) if int(nodes_df.iloc[i]["id"]) not in master_ids]
    target_rebel = int(round(len(remain) * float(np.clip(rebel_ratio, 0.0, 0.95))))
    if target_rebel > 0:
        ranked = sorted(remain, key=lambda nid: float(d_center[nid]), reverse=True)
        rebel_ids = set(ranked[:target_rebel])
    else:
        rebel_ids = set()
    out: dict[int, str] = {}
    for i in range(n):
        nid = int(nodes_df.iloc[i]["id"])
        if nid in master_ids:
            out[nid] = "master"
        elif nid in rebel_ids:
            out[nid] = "rebel"
        else:
            out[nid] = "follower"
    return out


def run_env_only(
    *,
    embeddings: np.ndarray,
    paths: list[str],
    output_dir: Path,
    k: int,
    T: int,
    M_cand: int,
    p_norm: float,
    p_comp: float,
    tau: float,
    wS: float,
    wM: float,
    s_gain: float,
    m_gain: float,
    mode: str,
    seed: int,
    game_top_k: int,
    game_aS: float,
    game_aM: float,
    game_aR: float,
    p_global_random: float,
    p_global_norm: float,
    p_global_dual: float,
    init_i_bonus: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, list[str], int, pd.DataFrame | None]:
    nodes, edges_citation, game_df = run_evolution(
        embeddings,
        paths,
        output_dir,
        k=k,
        T=T,
        M_cand=M_cand,
        p_norm=p_norm,
        p_comp=p_comp,
        tau=tau,
        wS=wS,
        wM=wM,
        s_gain=s_gain,
        m_gain=m_gain,
        mode=mode,
        seed=seed,
        game_top_k=game_top_k,
        aS=game_aS,
        aM=game_aM,
        aR=game_aR,
        p_global_random=p_global_random,
        p_global_norm=p_global_norm,
        p_global_dual=p_global_dual,
        init_i_bonus=init_i_bonus,
    )
    return nodes, edges_citation, game_df, embeddings, paths, len(paths), None


def run_env_plus_artist(
    *,
    embeddings: np.ndarray,
    paths: list[str],
    output_dir: Path,
    mode: str,
    T: int,
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
    game_aS: float,
    game_aM: float,
    game_aR: float,
    p_global_random: float,
    p_global_norm: float,
    p_global_dual: float,
    gen_engine: str,
    spawn_round_every: int,
    spawn_per_round: int,
    max_dynamic_nodes: int,
    gen_top_k: int,
    api_provider: str,
    api_model: str,
    api_fallback_models: list[str],
    api_endpoint: str,
    api_retry_max: int,
    api_retry_backoff_sec: float,
    decode_provider: str,
    decode_model: str,
    decode_endpoint: str,
    master_ratio: float,
    rebel_ratio: float,
    strategy_policy: str,
    gallery_size: int,
    master_visibility_boost: float,
    strict_llava_create: bool,
    strict_llava_decode: bool,
    feedback_i_scale: float,
    feedback_s_scale: float,
    feedback_m_scale: float,
    clip_model: str,
    clip_pretrained: str,
    device: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, list[str], int, pd.DataFrame | None]:
    if gen_engine == "none":
        raise ValueError("feedback_closed_loop=true requires generation enabled (set gen_engine to api or lora).")
    print(
        f"      [closed-loop] spawn_round_every={spawn_round_every} "
        f"spawn_per_round={spawn_per_round} max_dynamic_nodes={max_dynamic_nodes}"
    )
    cur_embeddings = np.asarray(embeddings, dtype=np.float32)
    cur_paths = list(paths)
    state = None
    spawn_rows = []
    gen_summary_rows = []
    round_cursor = 0
    while round_cursor <= T:
        should_spawn = (round_cursor % max(1, spawn_round_every) == 0)
        can_spawn = max(0, min(spawn_per_round, max_dynamic_nodes - len(cur_paths)))
        external_feedback_by_round = None
        if should_spawn and can_spawn > 0:
            if state is None:
                nodes_for_gen = pd.DataFrame({
                    "id": np.arange(len(cur_paths), dtype=int),
                    "path": cur_paths,
                    "S": np.ones(len(cur_paths), dtype=np.float64),
                    "M": np.ones(len(cur_paths), dtype=np.float64),
                    "I": np.ones(len(cur_paths), dtype=np.float64),
                })
            else:
                nodes_for_gen = pd.DataFrame({
                    "id": np.arange(len(cur_paths), dtype=int),
                    "path": cur_paths,
                    "S": np.asarray(state["S"], dtype=np.float64),
                    "M": np.asarray(state["M"], dtype=np.float64),
                    "I": np.asarray(state["I"], dtype=np.float64),
                })
            artist_types = infer_artist_types(
                nodes_for_gen,
                cur_embeddings[: len(nodes_for_gen)],
                master_ratio=master_ratio,
                rebel_ratio=rebel_ratio,
                strategy_policy=strategy_policy,
            )
            master_ids = {nid for nid, tp in artist_types.items() if tp == "master"}
            visible_gallery = sample_exposure_galleries(
                node_ids=nodes_for_gen["id"].to_numpy(dtype=int),
                influence=nodes_for_gen["I"].to_numpy(dtype=np.float64),
                gallery_size=gallery_size,
                rng=random.Random(seed + int(round_cursor) * 31),
                master_ids=master_ids,
                master_boost=master_visibility_boost,
            )
            one_df, one_summary = generate_round_images(
                output_dir=output_dir,
                nodes=nodes_for_gen,
                embeddings=cur_embeddings,
                mode=mode,
                round_idx=round_cursor,
                gen_per_round=can_spawn,
                gen_top_k=gen_top_k,
                gen_engine=gen_engine,
                api_provider=api_provider,
                api_model=api_model,
                api_fallback_models=api_fallback_models,
                api_endpoint=api_endpoint,
                api_retry_max=api_retry_max,
                api_retry_backoff_sec=api_retry_backoff_sec,
                visible_gallery=visible_gallery,
                artist_types=artist_types,
                decode_provider=decode_provider,
                decode_model=decode_model,
                decode_endpoint=decode_endpoint,
                strict_llava_create=strict_llava_create,
                strict_llava_decode=strict_llava_decode,
                feedback_i_scale=feedback_i_scale,
                feedback_s_scale=feedback_s_scale,
                feedback_m_scale=feedback_m_scale,
            )
            if not one_df.empty:
                gen_summary_rows.append(one_summary)
                new_image_paths = [Path(p) for p in one_df["image_path"].tolist()]
                new_emb = extract_embeddings(
                    new_image_paths,
                    model_name=clip_model,
                    pretrained=clip_pretrained,
                    device=device,
                    batch_size=16,
                )
                old_n = len(cur_paths)
                cur_embeddings = np.vstack([cur_embeddings, new_emb]).astype(np.float32)
                cur_paths.extend([str(p.resolve()) for p in new_image_paths])
                parent_ids = one_df["node_id"].astype(int).to_numpy()
                new_ids = np.arange(old_n, old_n + len(new_image_paths), dtype=int)
                for nid_new, nid_parent in zip(new_ids.tolist(), parent_ids.tolist()):
                    spawn_rows.append({
                        "round": int(round_cursor),
                        "new_node_id": int(nid_new),
                        "parent_node_id": int(nid_parent),
                        "path": cur_paths[nid_new],
                    })
                if state is not None:
                    S_old = np.asarray(state["S"], dtype=np.float64)
                    M_old = np.asarray(state["M"], dtype=np.float64)
                    I_old = np.asarray(state["I"], dtype=np.float64)
                    P_old = np.asarray(state["payoff_cum"], dtype=np.float64)
                    state["S"] = np.concatenate([S_old, np.clip(S_old[parent_ids] * 0.9, 0.5, None)])
                    state["M"] = np.concatenate([M_old, np.clip(M_old[parent_ids] * 0.9, 0.5, None)])
                    state["I"] = np.concatenate([I_old, np.clip(I_old[parent_ids] * 0.8, 0.5, None)])
                    state["payoff_cum"] = np.concatenate([P_old, np.zeros(len(parent_ids), dtype=np.float64)])
                fb_i = np.zeros(len(cur_paths), dtype=np.float64)
                fb_s = np.zeros(len(cur_paths), dtype=np.float64)
                fb_m = np.zeros(len(cur_paths), dtype=np.float64)
                for _, rr in one_df.iterrows():
                    src_id = int(rr["node_id"])
                    if 0 <= src_id < len(cur_paths):
                        fb_i[src_id] += float(rr.get("feedback_I", 0.0))
                        fb_s[src_id] += float(rr.get("feedback_S", 0.0))
                        fb_m[src_id] += float(rr.get("feedback_M", 0.0))
                external_feedback_by_round = {
                    int(round_cursor): {
                        "I_bonus": fb_i,
                        "S_bonus": fb_s,
                        "M_bonus": fb_m,
                    }
                }
        if round_cursor >= T:
            break
        next_cursor = min(T, round_cursor + max(1, spawn_round_every))
        delta = max(0, next_cursor - round_cursor)
        if delta > 0:
            state = run_evolution_stateful(
                embeddings=cur_embeddings,
                paths=cur_paths,
                round_start=round_cursor,
                round_count=delta,
                mode=mode,
                seed=seed,
                k=k,
                M_cand=M_cand,
                p_norm=p_norm,
                p_comp=p_comp,
                tau=tau,
                wS=wS,
                wM=wM,
                s_gain=s_gain,
                m_gain=m_gain,
                game_top_k=game_top_k,
                aS=game_aS,
                aM=game_aM,
                aR=game_aR,
                p_global_random=p_global_random,
                p_global_norm=p_global_norm,
                p_global_dual=p_global_dual,
                state=state,
                external_feedback_by_round=external_feedback_by_round,
            )
        round_cursor = next_cursor
    if state is None:
        raise RuntimeError("Closed-loop mode failed to produce evolution state.")
    nodes = pd.DataFrame({
        "id": np.arange(len(cur_paths), dtype=int),
        "path": cur_paths,
        "S": np.asarray(state["S"], dtype=np.float64),
        "M": np.asarray(state["M"], dtype=np.float64),
        "I": np.asarray(state["I"], dtype=np.float64),
    })
    edges_citation = pd.DataFrame(state.get("all_citations", []))
    game_df = pd.DataFrame(state.get("game_rows", []))
    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges_citation.to_csv(output_dir / "edges_citation.csv", index=False)
    game_df.to_csv(output_dir / "game_rounds.csv", index=False)
    if spawn_rows:
        pd.DataFrame(spawn_rows).to_csv(output_dir / "generated_nodes.csv", index=False)
    gen_summary = None
    if gen_summary_rows:
        gen_summary = pd.DataFrame(gen_summary_rows)
        gen_summary.to_csv(output_dir / "generation_rounds.csv", index=False)
    return nodes, edges_citation, game_df, cur_embeddings, cur_paths, len(cur_paths), gen_summary


def run_batch(
    *,
    project_dir: Path,
    config_path: Path,
    output_root: str | None,
    images: str | None,
    batch_kind: str,
    include_mixed_strategy: bool,
    repeats: int,
    T: int,
    spawn_round_every: int,
    spawn_per_round: int,
    gen_top_k: int,
) -> Path:
    base_cfg = load_config(config_path) if config_path.is_file() else {}
    batch_ts = now_compact()
    run_specs = build_batch_run_specs(
        batch_kind=batch_kind,
        include_mixed_strategy=include_mixed_strategy,
        repeats=repeats,
        batch_ts=batch_ts,
    )
    output_root_path = make_batch_root_dir(
        project_dir=project_dir,
        output_root=output_root,
        batch_kind=batch_kind,
        T=T,
        spawn_round_every=spawn_round_every,
        spawn_per_round=spawn_per_round,
        gen_top_k=gen_top_k,
        include_mixed_strategy=include_mixed_strategy,
        config_path=config_path,
    )
    output_root_path.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root_path / "batch_summary.csv"
    partial_csv = output_root_path / "batch_summary.partial.csv"
    status_json = output_root_path / "batch_status.json"

    strict_llava_create = bool(base_cfg.get("strict_llava_create", False))
    strict_llava_decode = bool(base_cfg.get("strict_llava_decode", False))
    results = []
    save_json(
        output_root_path / "batch_manifest.json",
        {
            "created_at": now_iso(),
            "config_path": str(config_path),
            "output_root": str(output_root_path),
            "batch_kind": batch_kind,
            "grid": "3x1" if batch_kind == "env_only" else ("3x4" if include_mixed_strategy else "3x3"),
            "repeats": int(repeats),
            "T": int(T),
            "spawn_round_every": int(spawn_round_every),
            "spawn_per_round": int(spawn_per_round),
            "gen_top_k": int(gen_top_k),
            "run_count": int(len(run_specs)),
            "runs": run_specs,
        },
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"

    for spec in run_specs:
        run_id = spec["run_id"]
        run_dir = output_root_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_cfg = dict(base_cfg)
        run_cfg.update(
            {
                "T": int(T),
                "obs_every": max(1, min(int(base_cfg.get("obs_every", 3)), int(T))),
                "mode": spec["mode"],
                "feedback_closed_loop": bool(batch_kind != "env_only"),
                "gen_engine": "api" if batch_kind != "env_only" else "none",
                "api_provider": str(base_cfg.get("api_provider", "openai")),
                "api_model": str(base_cfg.get("api_model", "llava")),
                "vlm_enabled": False,
                "spawn_round_every": int(spawn_round_every),
                "spawn_per_round": int(spawn_per_round),
                "gen_round_every": int(spawn_round_every),
                "gen_per_round": int(spawn_per_round),
                "gen_top_k": int(gen_top_k),
                "feedback_i_scale": float(base_cfg.get("feedback_i_scale", 2.0)),
                "feedback_s_scale": float(base_cfg.get("feedback_s_scale", 2.0)),
                "feedback_m_scale": float(base_cfg.get("feedback_m_scale", 2.0)),
                "strategy_policy": spec["strategy_policy"],
                "rebel_ratio": float(spec.get("rebel_ratio", base_cfg.get("rebel_ratio", 0.3))),
                "decode_provider": str(base_cfg.get("decode_provider", "openai")),
                "decode_model": str(base_cfg.get("decode_model", "llava")),
                "decode_endpoint": str(base_cfg.get("decode_endpoint", "")),
                "strict_llava_create": strict_llava_create,
                "strict_llava_decode": strict_llava_decode,
            }
        )
        run_cfg_path = run_dir / "run_config.json"
        save_json(run_cfg_path, run_cfg)
        cmd = [
            sys.executable,
            str((project_dir / "main.py").resolve()),
            "--pipeline",
            "env_only" if batch_kind == "env_only" else "env_plus_artist",
            "--config",
            str(run_cfg_path),
            "--output",
            str(run_dir),
            "--mode",
            str(spec["mode"]),
            "--device",
            "cpu",
            "--viz-engine",
            "matplotlib",
            "--gen-engine",
            "none" if batch_kind == "env_only" else "api",
        ]
        if images:
            cmd.extend(["--images", str(images)])
        print(f"\n===== Running {run_id} =====")
        print(" ".join(cmd))
        save_json(
            status_json,
            {
                "phase": "running",
                "current_run_id": run_id,
                "started_at": now_iso(),
                "completed_runs": int(len(results)),
                "total_runs": int(len(run_specs)),
            },
        )
        try:
            subprocess.run(cmd, cwd=project_dir, env=env, check=True)
            one = extract_batch_summary(run_dir / "metrics.csv", run_id)
        except subprocess.CalledProcessError as e:
            one = {"run_id": run_id, "status": f"failed_exit_{e.returncode}"}
        one["mode"] = spec["mode"]
        one["strategy_policy"] = spec["strategy_policy"]
        one["rebel_ratio"] = spec.get("rebel_ratio", np.nan)
        one["master_visibility_boost"] = float(run_cfg.get("master_visibility_boost", np.nan))
        one["finished_at"] = now_iso()
        results.append(one)
        partial_df = pd.DataFrame(results)
        save_csv_atomic(partial_csv, partial_df)
        save_json(
            status_json,
            {
                "phase": "running",
                "current_run_id": run_id,
                "last_finished_run_id": run_id,
                "last_finished_status": one.get("status", "unknown"),
                "updated_at": now_iso(),
                "completed_runs": int(len(results)),
                "total_runs": int(len(run_specs)),
            },
        )

    summary_df = pd.DataFrame(results)
    save_csv_atomic(summary_csv, summary_df)
    save_csv_atomic(partial_csv, summary_df)
    save_json(
        status_json,
        {
            "phase": "done",
            "updated_at": now_iso(),
            "completed_runs": int(len(results)),
            "total_runs": int(len(run_specs)),
            "summary_csv": str(summary_csv),
        },
    )
    print("\n===== Batch done =====")
    print(f"Summary -> {summary_csv}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    try:
        plot_dir = render_batch_compare_plots(output_root_path, batch_kind=batch_kind)
        if plot_dir is not None:
            print(f"Batch compare plots -> {plot_dir}")
    except Exception as e:
        print(f"[batch-viz][warn] failed to render batch compare plots: {e}")
    return output_root_path


def main():
    # Emit Python traceback on fatal signals (best effort for native crashes).
    faulthandler.enable(all_threads=True)
    load_dotenv_file(Path(__file__).resolve().parent / ".env")
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    parser = argparse.ArgumentParser(
        description="Renaissance style evolution MVP (1400-1600 patronage)"
    )
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--images", type=str, default=None, help="Override images_dir")
    parser.add_argument("--output", type=str, default=None, help="Override output_dir")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline_random", "norm_only", "dual_patronage"],
        default=None,
        help="baseline_random | norm_only | dual_patronage",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting (avoids matplotlib segfault on some systems)")
    parser.add_argument(
        "--viz-engine",
        type=str,
        choices=["auto", "matplotlib", "pil"],
        default=None,
        help="Plotting backend for step [5/5]. auto=prefer matplotlib, fallback PIL",
    )
    parser.add_argument(
        "--gen-engine",
        type=str,
        choices=["none", "api", "lora"],
        default=None,
        help="Optional generation stage backend at [6/6].",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["auto", "env_only", "env_plus_artist", "batch"],
        default="auto",
        help="Execution pipeline. auto picks env_plus_artist when feedback_closed_loop=true, else env_only.",
    )
    parser.add_argument("--batch-output-root", type=str, default=None, help="Optional. If empty, auto-create timestamped folder under runs_env_plus_artist/batch_runs/")
    parser.add_argument("--batch-kind", type=str, choices=["auto", "env_only", "env_plus_artist"], default="auto")
    parser.add_argument("--batch-include-mixed-strategy", action="store_true")
    parser.add_argument("--batch-repeats", type=int, default=1)
    parser.add_argument("--batch-T", type=int, default=40)
    parser.add_argument("--batch-spawn-round-every", type=int, default=2)
    parser.add_argument("--batch-spawn-per-round", type=int, default=4)
    parser.add_argument("--batch-gen-top-k", type=int, default=120)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path) if config_path.is_file() else {}
    images_dir = args.images or config.get("images_dir", "images")
    if args.mode is not None:
        config["mode"] = args.mode
    mode = config.get("mode", "dual_patronage")
    k = config.get("k", 15)
    T = config.get("T", 30)
    M_cand = config.get("M", 40)
    p_norm = config.get("p_norm", 0.2)
    p_comp = config.get("p_comp", 0.2)
    tau = config.get("tau", 0.0)
    wS = config.get("wS", 0.3)
    wM = config.get("wM", 0.3)
    s_gain = config.get("s_gain", 0.2)
    m_gain = config.get("m_gain", 0.2)
    obs_every = config.get("obs_every", 5)
    seed = config.get("seed", 42)
    clip_model = config.get("clip_model", "ViT-B-32")
    clip_pretrained = config.get("clip_pretrained", "openai")
    top_k_visible = config.get("top_k_visible", 80)
    k_means_k = config.get("k_means_k", 8)
    viz_engine = args.viz_engine or config.get("viz_engine", "auto")
    gen_engine = args.gen_engine or config.get("gen_engine", "none")
    gen_round_every = int(config.get("gen_round_every", obs_every))
    gen_per_round = int(config.get("gen_per_round", 8))
    gen_top_k = int(config.get("gen_top_k", 80))
    api_provider = str(config.get("api_provider", "mock"))
    api_model = str(config.get("api_model", "gpt-image-1"))
    api_fallback_models = config.get("api_fallback_models", ["openai/gpt-5-image"])
    api_endpoint = str(config.get("api_endpoint", "https://api.openai.com/v1/images/generations"))
    api_retry_max = int(config.get("api_retry_max", 3))
    api_retry_backoff_sec = float(config.get("api_retry_backoff_sec", 2.0))
    api_seed_policy = str(config.get("api_seed_policy", "round_node"))
    feedback_from_generated = bool(config.get("feedback_from_generated", False))
    feedback_closed_loop = bool(config.get("feedback_closed_loop", False))
    spawn_round_every = int(config.get("spawn_round_every", 10))
    spawn_per_round = int(config.get("spawn_per_round", 1))
    max_dynamic_nodes = int(config.get("max_dynamic_nodes", 400))
    vlm_enabled = bool(config.get("vlm_enabled", False))
    vlm_provider = str(config.get("vlm_provider", "mock"))
    vlm_model = str(config.get("vlm_model", "openai/gpt-4o-mini"))
    vlm_endpoint = str(config.get("vlm_endpoint", "https://openrouter.ai/api/v1/chat/completions"))
    lora_update_every = int(config.get("lora_update_every", 10))
    lora_steps = int(config.get("lora_steps", 300))
    lora_batch_size = int(config.get("lora_batch_size", 1))
    game_top_k = config.get("game_top_k", 80)
    game_aS = config.get("game_aS", 1.0)
    game_aM = config.get("game_aM", 1.0)
    game_aR = config.get("game_aR", 1.0)
    p_global_random = config.get("p_global_random", 0.1)
    p_global_norm = config.get("p_global_norm", 0.02)
    p_global_dual = config.get("p_global_dual", 0.2)
    gallery_size = int(config.get("gallery_size", 4))
    master_visibility_boost = float(config.get("master_visibility_boost", 3.0))
    master_ratio = float(config.get("master_ratio", 0.1))
    rebel_ratio = float(config.get("rebel_ratio", 0.3))
    strategy_policy = str(config.get("strategy_policy", "mixed"))
    decode_provider = str(config.get("decode_provider", "mock"))
    decode_model = str(config.get("decode_model", ""))
    decode_endpoint = str(config.get("decode_endpoint", ""))
    strict_llava_create = bool(config.get("strict_llava_create", False))
    strict_llava_decode = bool(config.get("strict_llava_decode", False))
    feedback_i_scale = float(config.get("feedback_i_scale", 1.0))
    feedback_s_scale = float(config.get("feedback_s_scale", 1.0))
    feedback_m_scale = float(config.get("feedback_m_scale", 1.0))
    selected_pipeline = str(args.pipeline or "auto").strip().lower()
    if selected_pipeline == "auto":
        selected_pipeline = "env_plus_artist" if feedback_closed_loop else "env_only"
    if selected_pipeline == "batch":
        batch_root = args.batch_output_root
        batch_kind = str(args.batch_kind or "auto").strip().lower()
        if batch_kind == "auto":
            batch_kind = "env_plus_artist" if feedback_closed_loop else "env_only"
        batch_dir = run_batch(
            project_dir=Path(__file__).resolve().parent,
            config_path=config_path,
            output_root=batch_root,
            images=args.images,
            batch_kind=batch_kind,
            include_mixed_strategy=bool(args.batch_include_mixed_strategy),
            repeats=int(args.batch_repeats),
            T=int(args.batch_T),
            spawn_round_every=int(args.batch_spawn_round_every),
            spawn_per_round=int(args.batch_spawn_per_round),
            gen_top_k=int(args.batch_gen_top_k),
        )
        print("Done. Batch outputs in", batch_dir.resolve(), flush=True)
        return
    if args.output is not None:
        output_dir = Path(args.output)
    else:
        output_dir = make_single_run_dir(
            project_dir=Path(__file__).resolve().parent,
            pipeline=selected_pipeline,
            mode=mode,
            strategy_policy=strategy_policy,
            T=T,
            spawn_round_every=spawn_round_every,
            spawn_per_round=spawn_per_round,
            gen_top_k=gen_top_k,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "run_manifest.json",
        {
            "created_at": now_iso(),
            "pipeline": selected_pipeline,
            "mode": mode,
            "strategy_policy": strategy_policy,
            "T": int(T),
            "seed": int(seed),
            "spawn_round_every": int(spawn_round_every),
            "spawn_per_round": int(spawn_per_round),
            "gen_top_k": int(gen_top_k),
            "config_path": str(config_path),
            "output_dir": str(output_dir.resolve()),
        },
    )
    init_i_bonus = None
    if feedback_from_generated:
        fb_path = output_dir / "generation_feedback.npy"
        if fb_path.is_file():
            init_i_bonus = np.load(fb_path)
            print(f"      loaded generation feedback bonus from {fb_path}")

    print("[1/5] Extracting CLIP embeddings...")
    embeddings, paths = run_embed(
        images_dir, output_dir,
        model_name=clip_model, pretrained=clip_pretrained, device=args.device,
    )
    n = len(paths)
    print(f"      {n} images -> shape {embeddings.shape}")

    print("[2/5] Building kNN graph (cosine)...")
    run_graph(embeddings, paths, output_dir, k=k)
    print("      nodes.csv, edges_knn.csv written")

    print(f"[3/5] Running evolution T={T} mode={mode}...")
    if selected_pipeline == "env_plus_artist":
        nodes, edges_citation, game_df, embeddings, paths, n, gen_summary = run_env_plus_artist(
            embeddings=embeddings,
            paths=paths,
            output_dir=output_dir,
            mode=mode,
            T=T,
            seed=seed,
            k=k,
            M_cand=M_cand,
            p_norm=p_norm,
            p_comp=p_comp,
            tau=tau,
            wS=wS,
            wM=wM,
            s_gain=s_gain,
            m_gain=m_gain,
            game_top_k=game_top_k,
            game_aS=game_aS,
            game_aM=game_aM,
            game_aR=game_aR,
            p_global_random=p_global_random,
            p_global_norm=p_global_norm,
            p_global_dual=p_global_dual,
            gen_engine=gen_engine,
            spawn_round_every=spawn_round_every,
            spawn_per_round=spawn_per_round,
            max_dynamic_nodes=max_dynamic_nodes,
            gen_top_k=gen_top_k,
            api_provider=api_provider,
            api_model=api_model,
            api_fallback_models=api_fallback_models,
            api_endpoint=api_endpoint,
            api_retry_max=api_retry_max,
            api_retry_backoff_sec=api_retry_backoff_sec,
            decode_provider=decode_provider,
            decode_model=decode_model,
            decode_endpoint=decode_endpoint,
            master_ratio=master_ratio,
            rebel_ratio=rebel_ratio,
            strategy_policy=strategy_policy,
            gallery_size=gallery_size,
            master_visibility_boost=master_visibility_boost,
            strict_llava_create=strict_llava_create,
            strict_llava_decode=strict_llava_decode,
            feedback_i_scale=feedback_i_scale,
            feedback_s_scale=feedback_s_scale,
            feedback_m_scale=feedback_m_scale,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            device=args.device,
        )
        feedback_closed_loop = True
    else:
        if selected_pipeline == "env_only" and feedback_closed_loop:
            print("      [pipeline] forcing env_only: overriding feedback_closed_loop=false")
            feedback_closed_loop = False
        nodes, edges_citation, game_df, embeddings, paths, n, gen_summary = run_env_only(
            embeddings=embeddings,
            paths=paths,
            output_dir=output_dir,
            k=k,
            T=T,
            M_cand=M_cand,
            p_norm=p_norm,
            p_comp=p_comp,
            tau=tau,
            wS=wS,
            wM=wM,
            s_gain=s_gain,
            m_gain=m_gain,
            mode=mode,
            seed=seed,
            game_top_k=game_top_k,
            game_aS=game_aS,
            game_aM=game_aM,
            game_aR=game_aR,
            p_global_random=p_global_random,
            p_global_norm=p_global_norm,
            p_global_dual=p_global_dual,
            init_i_bonus=init_i_bonus,
        )
    print(f"      edges_citation.csv: {len(edges_citation)} rows")

    obs_rounds = list(range(0, T + 1, obs_every))
    if obs_rounds[-1] != T:
        obs_rounds.append(T)
    print("[4/5] Computing metrics at rounds", obs_rounds, "...")
    metrics_df = run_metrics(
        embeddings, edges_citation, n_nodes=n,
        output_dir=output_dir, obs_rounds=obs_rounds,
        top_k_visible=top_k_visible, k_means_k=k_means_k, seed=seed,
    )
    # Merge evolutionary-game stats sampled at observation rounds.
    if game_df is not None and not game_df.empty:
        game_obs = game_df[game_df["round"].isin(obs_rounds)].copy()
        metrics_df = metrics_df.merge(game_obs, on="round", how="left")
    if feedback_closed_loop and gen_summary is not None and not gen_summary.empty:
        metrics_df = metrics_df.merge(gen_summary, on="round", how="left")
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print("      metrics.csv written")

    if args.no_plots:
        print("[5/5] Skipping plots (--no-plots).")
    else:
        print("[5/5] Writing plots...")
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
        try:
            subprocess.run(
                [sys.executable, "-X", "faulthandler", "-m", "src.viz", str(output_dir.resolve()), "--engine", viz_engine],
                cwd=Path(__file__).resolve().parent,
                env=env,
                timeout=120,
                check=True,
            )
            print(f"      plots -> {output_dir / 'plots'}")
        except subprocess.CalledProcessError as e:
            print(f"      plots failed (exit code {e.returncode}); metrics and data are in {output_dir}", flush=True)

    if gen_engine != "none" and not feedback_closed_loop:
        print(f"[6/6] Generate images engine={gen_engine} ...")
        if gen_engine == "api":
            gen_summary = run_generation(
                output_dir=output_dir,
                nodes=nodes,
                embeddings=embeddings,
                mode=mode,
                obs_rounds=obs_rounds,
                gen_round_every=gen_round_every,
                gen_per_round=gen_per_round,
                gen_top_k=gen_top_k,
                gen_engine="api",
                api_provider=api_provider,
                api_model=api_model,
                api_fallback_models=api_fallback_models,
                api_endpoint=api_endpoint,
                api_retry_max=api_retry_max,
                api_retry_backoff_sec=api_retry_backoff_sec,
                api_seed_policy=api_seed_policy,
                strict_llava_create=strict_llava_create,
            )
            if gen_summary is not None and not gen_summary.empty:
                metrics_df = pd.read_csv(output_dir / "metrics.csv")
                metrics_df = metrics_df.merge(gen_summary, on="round", how="left")
                metrics_df.to_csv(output_dir / "metrics.csv", index=False)
                print("      merged generation stats into metrics.csv")
                if not args.no_plots:
                    # Refresh plots that depend on generation stats (e.g., institution_vs_generation).
                    subprocess.run(
                        [sys.executable, "-X", "faulthandler", "-m", "src.viz", str(output_dir.resolve()), "--engine", viz_engine],
                        cwd=Path(__file__).resolve().parent,
                        env=env,
                        timeout=120,
                        check=True,
                    )
        elif gen_engine == "lora":
            ckpt_round = T if lora_update_every <= 0 else (T // lora_update_every) * lora_update_every
            ckpt_dir = output_dir / "lora" / "checkpoints" / f"round_{ckpt_round:04d}"
            subprocess.run(
                [
                    sys.executable, "-m", "src.lora_train",
                    "--data-dir", str((output_dir / "generated").resolve()),
                    "--output-dir", str(ckpt_dir.resolve()),
                    "--round", str(ckpt_round),
                    "--steps", str(lora_steps),
                    "--batch-size", str(lora_batch_size),
                ],
                cwd=Path(__file__).resolve().parent,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable, "-m", "src.lora_infer",
                    "--checkpoint-dir", str(ckpt_dir.resolve()),
                    "--output-dir", str((output_dir / "generated_lora" / f"round_{ckpt_round:04d}").resolve()),
                ],
                cwd=Path(__file__).resolve().parent,
                check=True,
            )
        print("      generation step completed")

    if vlm_enabled:
        print("[VLM] Evaluating generated images...")
        vlm_df = run_vlm_eval(
            output_dir=output_dir,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            vlm_endpoint=vlm_endpoint,
        )
        if vlm_df is not None and not vlm_df.empty:
            metrics_df = pd.read_csv(output_dir / "metrics.csv")
            metrics_df = metrics_df.merge(vlm_df, on="round", how="left")
            metrics_df.to_csv(output_dir / "metrics.csv", index=False)
            print("      merged VLM stats into metrics.csv")
            if not args.no_plots:
                subprocess.run(
                    [sys.executable, "-X", "faulthandler", "-m", "src.viz", str(output_dir.resolve()), "--engine", viz_engine],
                    cwd=Path(__file__).resolve().parent,
                    env=env if "env" in locals() else os.environ.copy(),
                    timeout=120,
                    check=True,
                )
        else:
            print("      no generated rounds found for VLM evaluation")
    print("Done. Outputs in", output_dir.resolve(), flush=True)


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    main()
