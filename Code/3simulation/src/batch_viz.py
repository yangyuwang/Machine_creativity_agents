"""
Batch-level comparison plots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ENV_ORDER = ["baseline_random", "norm_only", "dual_patronage"]
STRATEGY_ORDER = ["imitation_only", "differentiation_only", "self_consistency_only"]


def _safe_float(v: object) -> float:
    vv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if pd.isna(vv):
        return float("nan")
    return float(vv)


def _draw_heatmap(ax: plt.Axes, mat: pd.DataFrame, title: str, cmap: str = "viridis") -> None:
    arr = mat.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(list(mat.columns), rotation=10)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(list(mat.index))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_env_only(summary: pd.DataFrame, batch_dir: Path, out_dir: Path) -> None:
    dd = summary.copy()
    dd["mode"] = dd["mode"].astype(str)
    dd = dd.set_index("mode").reindex(ENV_ORDER).reset_index()

    metrics = [
        ("modularity", "Modularity"),
        ("cross_cluster_citation_rate", "Cross-Cluster Citation Rate"),
        ("in_degree_top10_share", "Top10 In-degree Share"),
        ("topk_turnover_rate", "TopK Turnover Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = pd.to_numeric(dd[col], errors="coerce")
        ax.bar(dd["mode"], vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=10)
        for i, v in enumerate(vals):
            if pd.notna(v):
                ax.text(i, float(v), f"{float(v):.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Env-Only Batch (3x1) Final Metrics Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "env_only_3x1_final_metrics.png", dpi=180)
    plt.close(fig)

    line_cols = [
        ("modularity", "Modularity over Rounds"),
        ("cross_cluster_citation_rate", "Cross-Cluster Citation over Rounds"),
        ("in_degree_top10_share", "Top10 Share over Rounds"),
        ("topk_turnover_rate", "TopK Turnover over Rounds"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for mode, color in zip(ENV_ORDER, colors):
        row = dd[dd["mode"] == mode]
        if row.empty:
            continue
        run_id = str(row.iloc[0]["run_id"])
        mpath = batch_dir / run_id / "metrics.csv"
        if not mpath.exists():
            continue
        rdf = pd.read_csv(mpath).sort_values("round")
        for ax, (col, title) in zip(axes.ravel(), line_cols):
            y = pd.to_numeric(rdf[col], errors="coerce")
            ax.plot(rdf["round"], y, marker="o", linewidth=1.8, label=mode, color=color)
            ax.set_title(title)
            ax.set_xlabel("round")
    fig.legend(ENV_ORDER, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Env-Only Batch (3x1) Dynamics Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "env_only_3x1_dynamics_overlay.png", dpi=180)
    plt.close(fig)

    lines = ["# Env-only 3x1 comparison"]
    for _, r in dd.iterrows():
        lines.append(
            f"- {r['mode']}: modularity={float(r['modularity']):.3f}, "
            f"cross_citation={float(r['cross_cluster_citation_rate']):.3f}, "
            f"top10_share={float(r['in_degree_top10_share']):.3f}, "
            f"turnover={float(r['topk_turnover_rate']):.3f}"
        )
    (out_dir / "env_only_3x1_brief.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_env_plus_artist(summary: pd.DataFrame, batch_dir: Path, out_dir: Path) -> None:
    dd = summary.copy()
    dd["mode"] = dd["mode"].astype(str)
    dd["strategy_policy"] = dd["strategy_policy"].astype(str)
    dd = dd[dd["mode"].isin(ENV_ORDER) & dd["strategy_policy"].isin(STRATEGY_ORDER)].copy()
    if dd.empty:
        return

    key_metrics = [
        ("modularity", "Modularity"),
        ("cross_cluster_citation_rate", "Cross-Cluster Citation"),
        ("in_degree_top10_share", "Top10 In-degree Share"),
        ("topk_turnover_rate", "TopK Turnover"),
    ]

    # Figure 1: 4 heatmaps, each is environment x strategy.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (metric, title) in zip(axes.ravel(), key_metrics):
        table = (
            dd.pivot_table(index="mode", columns="strategy_policy", values=metric, aggfunc="mean")
            .reindex(index=ENV_ORDER, columns=STRATEGY_ORDER)
        )
        _draw_heatmap(ax, table, title=f"{title} (final)")
    fig.suptitle("Env+Artist Batch (3x3) Heatmap Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "env_plus_artist_3x3_heatmaps.png", dpi=180)
    plt.close(fig)

    # Figure 2: grouped bars for final metrics.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(STRATEGY_ORDER))
    bar_w = 0.25
    env_colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, (metric, title) in zip(axes.ravel(), key_metrics):
        for idx, (env, color) in enumerate(zip(ENV_ORDER, env_colors)):
            vals = []
            for st in STRATEGY_ORDER:
                hit = dd[(dd["mode"] == env) & (dd["strategy_policy"] == st)]
                vals.append(_safe_float(hit.iloc[0][metric]) if not hit.empty else np.nan)
            ax.bar(x + (idx - 1) * bar_w, vals, width=bar_w, label=env, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_ORDER, rotation=10)
        ax.set_title(title)
    fig.legend(ENV_ORDER, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Env+Artist Batch (3x3) Final Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "env_plus_artist_3x3_grouped_bars.png", dpi=180)
    plt.close(fig)

    # Figure 3: round dynamics overlay across all 9 cells.
    line_cols = [
        ("modularity", "Modularity over Rounds"),
        ("cross_cluster_citation_rate", "Cross-Cluster Citation over Rounds"),
        ("in_degree_top10_share", "Top10 Share over Rounds"),
        ("topk_turnover_rate", "TopK Turnover over Rounds"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for env in ENV_ORDER:
        for st in STRATEGY_ORDER:
            hit = dd[(dd["mode"] == env) & (dd["strategy_policy"] == st)]
            if hit.empty:
                continue
            run_id = str(hit.iloc[0]["run_id"])
            mpath = batch_dir / run_id / "metrics.csv"
            if not mpath.exists():
                continue
            rdf = pd.read_csv(mpath).sort_values("round")
            label = f"{env}|{st}"
            for ax, (col, title) in zip(axes.ravel(), line_cols):
                y = pd.to_numeric(rdf[col], errors="coerce")
                ax.plot(rdf["round"], y, linewidth=1.5, alpha=0.9, label=label)
                ax.set_title(title)
                ax.set_xlabel("round")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle("Env+Artist Batch (3x3) Dynamics Overlay", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "env_plus_artist_3x3_dynamics_overlay.png", dpi=180)
    plt.close(fig)

    # Brief markdown: highlight best cell for each key metric.
    lines = ["# Env+Artist 3x3 comparison", ""]
    for metric, title in key_metrics:
        cur = dd[["mode", "strategy_policy", metric]].copy()
        cur[metric] = pd.to_numeric(cur[metric], errors="coerce")
        cur = cur.dropna(subset=[metric])
        if cur.empty:
            continue
        top = cur.sort_values(metric, ascending=False).iloc[0]
        bottom = cur.sort_values(metric, ascending=True).iloc[0]
        lines.append(
            f"- {title}: best={top['mode']} + {top['strategy_policy']} ({float(top[metric]):.3f}), "
            f"worst={bottom['mode']} + {bottom['strategy_policy']} ({float(bottom[metric]):.3f})"
        )
    (out_dir / "env_plus_artist_3x3_brief.md").write_text("\n".join(lines), encoding="utf-8")


def render_batch_compare_plots(batch_dir: Path, batch_kind: str) -> Path | None:
    summary_path = batch_dir / "batch_summary.csv"
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return None
    out_dir = batch_dir / "plots_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = str(batch_kind or "").strip().lower()
    if kind == "env_only":
        _plot_env_only(summary, batch_dir=batch_dir, out_dir=out_dir)
    elif kind == "env_plus_artist":
        _plot_env_plus_artist(summary, batch_dir=batch_dir, out_dir=out_dir)
    return out_dir

