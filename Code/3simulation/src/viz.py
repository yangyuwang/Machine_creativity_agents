"""
UMAP by cluster / by influence; metrics curves (incl. modularity and classicization).
Uses Pillow (PIL) only to avoid matplotlib segfault on some systems (e.g. macOS + numba/loky).
"""
import argparse
from pathlib import Path
from typing import Optional

import faulthandler
import numpy as np
import pandas as pd

_MPL_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

def _get_font(size: int = 12):
    if ImageFont is None:
        return None
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return None


def run_all_matplotlib(
    output_dir: Path,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    influence: Optional[np.ndarray] = None,
    metrics_df: Optional[pd.DataFrame] = None,
) -> None:
    if not _MPL_OK:
        raise RuntimeError("matplotlib backend not available in this environment")
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # UMAP by cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap="tab20", s=12, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_title("UMAP by cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(plot_dir / "umap_cluster.png", dpi=140)
    plt.close(fig)

    # UMAP by influence
    if influence is not None:
        inf = np.asarray(influence, dtype=np.float64)
        finite = np.isfinite(inf)
        if finite.any():
            hi = np.nanpercentile(inf[finite], 99)
            inf = np.clip(inf, np.nanmin(inf[finite]), hi)
        size = 8 + 28 * (inf - np.nanmin(inf)) / (np.nanmax(inf) - np.nanmin(inf) + 1e-8)
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=inf, s=size, cmap="viridis", alpha=0.8)
        plt.colorbar(sc, ax=ax, label="Influence")
        ax.set_title("UMAP by influence (I)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        fig.savefig(plot_dir / "umap_influence.png", dpi=140)
        plt.close(fig)

    if metrics_df is None or metrics_df.empty:
        return

    # metrics curves
    df = metrics_df.copy()
    r = pd.to_numeric(df["round"], errors="coerce").to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    series = [
        ("n_clusters", "Number of clusters"),
        ("modularity", "Modularity (citation)"),
        ("in_degree_top10_share", "Top10% in-degree share"),
        ("gini_in_degree", "Gini(in-degree)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), series):
        if col not in df.columns:
            ax.set_title(title)
            ax.text(0.2, 0.5, f"Missing: {col}", transform=ax.transAxes)
            continue
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        ax.plot(r, v, "o-", lw=2, ms=4)
        ax.set_title(title)
        ax.set_xlabel("Round")
        finite = np.isfinite(v)
        if finite.any():
            vmin, vmax = np.nanmin(v[finite]), np.nanmax(v[finite])
            ax.text(0.02, 0.94, f"vmin={vmin:.3f}\nvmax={vmax:.3f}", transform=ax.transAxes, fontsize=8, va="top")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "metrics_curves.png", dpi=140)
    plt.close(fig)

    # intra/inter
    if "intra_cluster_mean_dist" in df.columns and "inter_cluster_centroid_dist" in df.columns:
        intra = pd.to_numeric(df["intra_cluster_mean_dist"], errors="coerce").to_numpy(dtype=np.float64)
        inter = pd.to_numeric(df["inter_cluster_centroid_dist"], errors="coerce").to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(r, intra, "o-", label="Intra-cluster mean dist", lw=2)
        ax.plot(r, inter, "s-", label="Inter-cluster centroid dist", lw=2)
        ax.set_title("Clustering distances")
        ax.set_xlabel("Round")
        ax.set_ylabel("Distance")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "intra_inter_dist.png", dpi=140)
        plt.close(fig)

    # separation ratio
    if "separation_ratio" in df.columns:
        ratio = pd.to_numeric(df["separation_ratio"], errors="coerce").to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r, ratio, "o-", color="#2ca02c", lw=2)
        ax.set_title("Separation ratio (inter / intra)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Ratio")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "separation_ratio.png", dpi=140)
        plt.close(fig)

    # game dynamics
    game_cols = [
        "share_conform_topK",
        "payoff_conform_mean",
        "payoff_diff_mean",
        "payoff_conform_cum_mean",
        "payoff_diff_cum_mean",
    ]
    if all(c in df.columns for c in game_cols):
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        share = pd.to_numeric(df["share_conform_topK"], errors="coerce").to_numpy(dtype=np.float64)
        axes[0].plot(r, share, "o-", color="#1f77b4", lw=2, ms=4, label="share_conform_topK")
        axes[0].set_ylabel("Share")
        axes[0].set_title("TopK strategy composition")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        p_conf = pd.to_numeric(df["payoff_conform_mean"], errors="coerce").to_numpy(dtype=np.float64)
        p_diff = pd.to_numeric(df["payoff_diff_mean"], errors="coerce").to_numpy(dtype=np.float64)
        axes[1].plot(r, p_conf, "o-", lw=2, ms=4, label="payoff_conform_mean")
        axes[1].plot(r, p_diff, "s-", lw=2, ms=4, label="payoff_diff_mean")
        axes[1].set_ylabel("Instant payoff")
        axes[1].set_title("Instant payoff by strategy")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        c_conf = pd.to_numeric(df["payoff_conform_cum_mean"], errors="coerce").to_numpy(dtype=np.float64)
        c_diff = pd.to_numeric(df["payoff_diff_cum_mean"], errors="coerce").to_numpy(dtype=np.float64)
        axes[2].plot(r, c_conf, "o-", lw=2, ms=4, label="payoff_conform_cum_mean")
        axes[2].plot(r, c_diff, "s-", lw=2, ms=4, label="payoff_diff_cum_mean")
        axes[2].set_ylabel("Cumulative payoff")
        axes[2].set_xlabel("Round")
        axes[2].set_title("Cumulative payoff by strategy")
        axes[2].grid(alpha=0.25)
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "game_dynamics.png", dpi=140)
        plt.close(fig)

    # cross-cluster citation rate
    if "cross_cluster_citation_rate" in df.columns:
        y = pd.to_numeric(df["cross_cluster_citation_rate"], errors="coerce").to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(r, y, "o-", lw=2, color="#d62728")
        ax.set_title("Cross-cluster citation rate")
        ax.set_xlabel("Round")
        ax.set_ylabel("Rate")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "citation_cross_rate.png", dpi=140)
        plt.close(fig)

    # all-vs-visible cluster counts
    if "clusters_all" in df.columns and "clusters_visible" in df.columns:
        ca = pd.to_numeric(df["clusters_all"], errors="coerce").to_numpy(dtype=np.float64)
        cv = pd.to_numeric(df["clusters_visible"], errors="coerce").to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(r, ca, "o-", lw=2, label="clusters_all")
        ax.plot(r, cv, "s-", lw=2, label="clusters_visible")
        ax.set_title("Clusters: all vs visible")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cluster count")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "clusters_all_vs_visible.png", dpi=140)
        plt.close(fig)

    if "topk_turnover_rate" in df.columns:
        y = pd.to_numeric(df["topk_turnover_rate"], errors="coerce").to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(r, y, "o-", lw=2, color="#8c564b")
        ax.set_title("Top-K turnover rate")
        ax.set_xlabel("Round")
        ax.set_ylabel("Turnover")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "topk_turnover_rate.png", dpi=140)
        plt.close(fig)

    # institution vs generation linkage (phase-3)
    gen_cols = ["generated_count", "rgb_std_mean", "center_deviation_proxy"]
    if all(c in df.columns for c in gen_cols):
        gcount = pd.to_numeric(df["generated_count"], errors="coerce").to_numpy(dtype=np.float64)
        gdiv = pd.to_numeric(df["rgb_std_mean"], errors="coerce").to_numpy(dtype=np.float64)
        gdev = pd.to_numeric(df["center_deviation_proxy"], errors="coerce").to_numpy(dtype=np.float64)
        mod = pd.to_numeric(df["modularity"], errors="coerce").to_numpy(dtype=np.float64) if "modularity" in df.columns else None
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
        axes[0].plot(r, gcount, "o-", lw=2, color="#1f77b4", label="generated_count")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Generation activity")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        axes[1].plot(r, gdiv, "o-", lw=2, color="#2ca02c", label="rgb_std_mean")
        axes[1].plot(r, gdev, "s-", lw=2, color="#9467bd", label="center_deviation_proxy")
        axes[1].set_ylabel("Gen style proxy")
        axes[1].set_title("Generation diversity and center deviation")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        if mod is not None:
            axes[2].plot(r, mod, "o-", lw=2, color="#d62728", label="modularity")
            axes[2].set_title("Institution metric")
        else:
            axes[2].plot(r, gdev, "o-", lw=2, color="#9467bd", label="center_deviation_proxy")
            axes[2].set_title("Institution metric unavailable (showing center deviation)")
        axes[2].set_xlabel("Round")
        axes[2].set_ylabel("Value")
        axes[2].grid(alpha=0.25)
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "institution_vs_generation.png", dpi=140)
        plt.close(fig)

    vlm_cols = [
        "vlm_style_conformity_mean",
        "vlm_novelty_score_mean",
        "vlm_prompt_alignment_mean",
        "vlm_craft_score_mean",
    ]
    if all(c in df.columns for c in vlm_cols):
        v1 = pd.to_numeric(df["vlm_style_conformity_mean"], errors="coerce").to_numpy(dtype=np.float64)
        v2 = pd.to_numeric(df["vlm_novelty_score_mean"], errors="coerce").to_numpy(dtype=np.float64)
        v3 = pd.to_numeric(df["vlm_prompt_alignment_mean"], errors="coerce").to_numpy(dtype=np.float64)
        v4 = pd.to_numeric(df["vlm_craft_score_mean"], errors="coerce").to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(r, v1, "o-", lw=2, label="style_conformity")
        ax.plot(r, v2, "s-", lw=2, label="novelty_score")
        ax.plot(r, v3, "^-", lw=2, label="prompt_alignment")
        ax.plot(r, v4, "d-", lw=2, label="craft_score")
        ax.set_title("VLM evaluation dynamics")
        ax.set_xlabel("Round")
        ax.set_ylabel("Score (0-1)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "vlm_dynamics.png", dpi=140)
        plt.close(fig)

# Simple color palette (tab20-like) and viridis-like for continuous
_TAB20 = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
]


def _scale_to_margin(x: np.ndarray, y: np.ndarray, w: int, h: int, margin: int = 40) -> tuple:
    """Scale x,y to fit in (margin, margin) .. (w-margin, h-margin). Return (x_px, y_px)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1
    xp = margin + (x - xmin) / (xmax - xmin) * (w - 2 * margin)
    yp = (h - margin) - (y - ymin) / (ymax - ymin) * (h - 2 * margin)  # y flip
    return xp, yp


def _viridis(t: np.ndarray) -> np.ndarray:
    """t in [0,1] -> RGB (N,3) uint8. Approximate viridis."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0, 1)
    # piecewise linear approx
    r = np.where(t < 0.25, 0.0, np.where(t < 0.5, (t - 0.25) * 4, np.where(t < 0.75, 1.0, 1.0 - (t - 0.75) * 4)))
    g = np.where(t < 0.25, t * 4, np.where(t < 0.5, 1.0, np.where(t < 0.75, 1.0 - (t - 0.5) * 4, (t - 0.75) * 4)))
    b = np.where(t < 0.5, 0.25 + t * 4, np.where(t < 0.75, 1.0, 1.0 - (t - 0.75) * 4))
    return (np.column_stack([r, g, b]) * 255).astype(np.uint8)


def plot_umap_cluster(coords_2d: np.ndarray, labels: np.ndarray, path: Path, title: str = "UMAP by cluster") -> None:
    if Image is None or ImageDraw is None:
        return
    w, h = 960, 720
    margin = 50
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(14)
    kw = {} if font is None else {"font": font}
    x, y = coords_2d[:, 0], coords_2d[:, 1]
    xp, yp = _scale_to_margin(x, y, w, h, margin=margin)
    lbls = np.asarray(labels, dtype=int)
    for i in range(len(xp)):
        c = _TAB20[lbls[i] % len(_TAB20)]
        r = 4
        draw.ellipse((xp[i] - r, yp[i] - r, xp[i] + r, yp[i] + r), fill=c, outline=(80, 80, 80))
    draw.text((20, 12), title, fill=(0, 0, 0), **kw)
    draw.text((w // 2 - 25, h - 28), "UMAP 1", fill=(0, 0, 0), **kw)
    draw.text((12, h // 2 - 30), "UMAP 2", fill=(0, 0, 0), **kw)
    img.save(path, "PNG")


def plot_umap_influence(coords_2d: np.ndarray, influence: np.ndarray, path: Path, title: str = "UMAP by influence (I)") -> None:
    if Image is None or ImageDraw is None:
        return
    w, h = 960, 720
    margin = 50
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(14)
    kw = {} if font is None else {"font": font}
    x, y = coords_2d[:, 0], coords_2d[:, 1]
    xp, yp = _scale_to_margin(x, y, w, h, margin=margin)
    inf = np.asarray(influence, dtype=np.float64)
    inf = np.clip(inf, None, np.nanpercentile(inf, 99) if np.any(np.isfinite(inf)) else 1.0)
    mn, mx = np.nanmin(inf), np.nanmax(inf)
    if mx <= mn:
        mx = mn + 1
    t = (inf - mn) / (mx - mn)
    colors = _viridis(t)
    sizes = (2 + 8 * t).astype(int)
    for i in range(len(xp)):
        r = max(1, sizes[i])
        c = tuple(int(colors[i, j]) for j in range(3))
        draw.ellipse((xp[i] - r, yp[i] - r, xp[i] + r, yp[i] + r), fill=c, outline=(60, 60, 60))
    draw.text((20, 12), title, fill=(0, 0, 0), **kw)
    draw.text((w // 2 - 25, h - 28), "UMAP 1", fill=(0, 0, 0), **kw)
    draw.text((12, h // 2 - 30), "UMAP 2", fill=(0, 0, 0), **kw)
    img.save(path, "PNG")


def plot_metrics_curves(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    w, h = 960, 720
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    m = 50
    cw, ch = (w - 3 * m) // 2, (h - 3 * m) // 2
    titles = ["Number of clusters", "Modularity", "In-degree top 10% share", "Gini(in-degree)"]
    cols = ["n_clusters", "modularity", "in_degree_top10_share", "gini_in_degree"]
    for idx, (col, title) in enumerate(zip(cols, titles)):
        row, col_idx = idx // 2, idx % 2
        x0, y0 = m + col_idx * (cw + m), m + row * (ch + m)
        draw.rectangle([x0, y0, x0 + cw, y0 + ch], outline=(200, 200, 200), width=1)
        r = metrics_df["round"].values
        v = metrics_df[col].values.astype(np.float64)
        v = np.nan_to_num(v, nan=0.0)
        if len(r) < 2:
            draw.text((x0 + 10, y0 + 8), title[:24], fill=(0, 0, 0), **kw)
            continue
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if vmax <= vmin:
            vmax = vmin + 1
        pad = 24
        xp = x0 + pad + (r - r.min()) / (r.max() - r.min() + 1e-8) * (cw - pad - 10)
        yp = y0 + ch - pad - (v - vmin) / (vmax - vmin) * (ch - pad - 10)
        pts = [(xp[i], yp[i]) for i in range(len(r))]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(31, 119, 180), width=2)
        draw.text((x0 + 8, y0 + 4), title[:22], fill=(0, 0, 0), **kw)
        draw.text((x0 + cw // 2 - 18, y0 + ch - 18), "Round", fill=(0, 0, 0), **kw)
        draw.text((x0 + 8, y0 + ch - 18), f"vmin={vmin:.3f}", fill=(60, 60, 60), **kw)
        draw.text((x0 + cw - 86, y0 + 4), f"vmax={vmax:.3f}", fill=(60, 60, 60), **kw)
    img.save(path, "PNG")


def plot_intra_inter(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    margin = 56
    top_legend = 38
    plot_left, plot_right = margin, w - margin
    plot_top, plot_bottom = top_legend + 8, h - margin
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=(180, 180, 180), width=1)
    r = metrics_df["round"].values
    a = metrics_df["intra_cluster_mean_dist"].values.astype(np.float64)
    b = metrics_df["inter_cluster_centroid_dist"].values.astype(np.float64)
    a, b = np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0)
    xmin, xmax = r.min(), r.max()
    if xmax <= xmin:
        xmax = xmin + 1
    yall = np.concatenate([a, b])
    ymin, ymax = np.nanmin(yall), np.nanmax(yall)
    if ymax <= ymin:
        ymax = ymin + 1
    xp = plot_left + (r - xmin) / (xmax - xmin) * (plot_w - 4)
    ypa = plot_bottom - (a - ymin) / (ymax - ymin) * (plot_h - 4)
    ypb = plot_bottom - (b - ymin) / (ymax - ymin) * (plot_h - 4)
    for i in range(len(r) - 1):
        draw.line([(xp[i], ypa[i]), (xp[i + 1], ypa[i + 1])], fill=(31, 119, 180), width=2)
        draw.line([(xp[i], ypb[i]), (xp[i + 1], ypb[i + 1])], fill=(255, 127, 14), width=2)
    draw.text((20, 10), "Clustering distances", fill=(0, 0, 0), **kw)
    draw.text((w // 2 - 22, h - 24), "Round", fill=(0, 0, 0), **kw)
    draw.text((14, plot_top + plot_h // 2 - 24), "Distance", fill=(0, 0, 0), **kw)
    leg_y = 14
    draw.line([(plot_left, leg_y), (plot_left + 32, leg_y)], fill=(31, 119, 180), width=3)
    draw.text((plot_left + 38, leg_y - 5), "Intra mean", fill=(0, 0, 0), **kw)
    draw.line([(plot_left + 140, leg_y), (plot_left + 172, leg_y)], fill=(255, 127, 14), width=3)
    draw.text((plot_left + 178, leg_y - 5), "Inter centroid", fill=(0, 0, 0), **kw)
    img.save(path, "PNG")


def plot_separation_ratio(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    if "separation_ratio" not in metrics_df.columns:
        return
    w, h = 640, 420
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    margin = 56
    left, right = margin, w - margin
    top, bottom = 48, h - margin
    draw.rectangle([left, top, right, bottom], outline=(180, 180, 180), width=1)
    r = pd.to_numeric(metrics_df["round"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(metrics_df["separation_ratio"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(r) & np.isfinite(y)
    if mask.sum() >= 2:
        rr, yy = r[mask], y[mask]
        xmin, xmax = rr.min(), rr.max()
        ymin, ymax = yy.min(), yy.max()
        if xmax <= xmin:
            xmax = xmin + 1
        if ymax <= ymin:
            ymax = ymin + 1
        xp = left + (rr - xmin) / (xmax - xmin) * (right - left - 4)
        yp = bottom - (yy - ymin) / (ymax - ymin) * (bottom - top - 4)
        for i in range(len(rr) - 1):
            draw.line([(xp[i], yp[i]), (xp[i + 1], yp[i + 1])], fill=(44, 160, 44), width=2)
        draw.text((left + 8, bottom - 18), f"vmin={ymin:.3f}", fill=(60, 60, 60), **kw)
        draw.text((right - 90, top + 6), f"vmax={ymax:.3f}", fill=(60, 60, 60), **kw)
    draw.text((18, 12), "Separation ratio (inter / intra)", fill=(0, 0, 0), **kw)
    draw.text((w // 2 - 22, h - 24), "Round", fill=(0, 0, 0), **kw)
    draw.text((10, (top + bottom) // 2 - 14), "Ratio", fill=(0, 0, 0), **kw)
    img.save(path, "PNG")


def plot_game_dynamics_pil(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    needed = [
        "share_conform_topK",
        "payoff_conform_mean",
        "payoff_diff_mean",
        "payoff_conform_cum_mean",
        "payoff_diff_cum_mean",
    ]
    if not all(c in metrics_df.columns for c in needed):
        return
    w, h = 760, 900
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    r = pd.to_numeric(metrics_df["round"], errors="coerce").to_numpy(dtype=np.float64)
    panels = [
        ("TopK composition", [("share_conform_topK", (31, 119, 180))]),
        ("Instant payoff", [("payoff_conform_mean", (31, 119, 180)), ("payoff_diff_mean", (255, 127, 14))]),
        ("Cumulative payoff", [("payoff_conform_cum_mean", (31, 119, 180)), ("payoff_diff_cum_mean", (255, 127, 14))]),
    ]
    panel_h = 260
    for p_idx, (title, series) in enumerate(panels):
        x0, y0 = 60, 40 + p_idx * (panel_h + 20)
        x1, y1 = w - 40, y0 + panel_h
        draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)
        vals = []
        for col, _ in series:
            vals.append(pd.to_numeric(metrics_df[col], errors="coerce").to_numpy(dtype=np.float64))
        yall = np.concatenate([v[np.isfinite(v)] for v in vals if np.isfinite(v).any()]) if vals else np.array([])
        if yall.size == 0:
            draw.text((x0 + 8, y0 + 6), f"{title} (no data)", fill=(0, 0, 0), **kw)
            continue
        xmin, xmax = np.nanmin(r), np.nanmax(r)
        ymin, ymax = np.nanmin(yall), np.nanmax(yall)
        if xmax <= xmin:
            xmax = xmin + 1
        if ymax <= ymin:
            ymax = ymin + 1
        draw.text((x0 + 8, y0 + 6), title, fill=(0, 0, 0), **kw)
        for col, color in series:
            v = pd.to_numeric(metrics_df[col], errors="coerce").to_numpy(dtype=np.float64)
            mask = np.isfinite(r) & np.isfinite(v)
            if mask.sum() < 2:
                continue
            rr, vv = r[mask], v[mask]
            xp = x0 + 30 + (rr - xmin) / (xmax - xmin) * (x1 - x0 - 50)
            yp = y1 - 20 - (vv - ymin) / (ymax - ymin) * (y1 - y0 - 40)
            for i in range(len(xp) - 1):
                draw.line([(xp[i], yp[i]), (xp[i + 1], yp[i + 1])], fill=color, width=2)
    img.save(path, "PNG")


def plot_cross_cluster_rate_pil(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    if "cross_cluster_citation_rate" not in metrics_df.columns:
        return
    w, h = 700, 420
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    r = pd.to_numeric(metrics_df["round"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(metrics_df["cross_cluster_citation_rate"], errors="coerce").to_numpy(dtype=np.float64)
    left, right, top, bottom = 56, w - 40, 48, h - 52
    draw.rectangle([left, top, right, bottom], outline=(180, 180, 180), width=1)
    mask = np.isfinite(r) & np.isfinite(y)
    if mask.sum() >= 2:
        rr, yy = r[mask], y[mask]
        xmin, xmax = rr.min(), rr.max()
        ymin, ymax = yy.min(), yy.max()
        if xmax <= xmin:
            xmax = xmin + 1
        if ymax <= ymin:
            ymax = ymin + 1
        xp = left + (rr - xmin) / (xmax - xmin) * (right - left - 4)
        yp = bottom - (yy - ymin) / (ymax - ymin) * (bottom - top - 4)
        for i in range(len(rr) - 1):
            draw.line([(xp[i], yp[i]), (xp[i + 1], yp[i + 1])], fill=(214, 39, 40), width=2)
    draw.text((16, 12), "Cross-cluster citation rate", fill=(0, 0, 0), **kw)
    draw.text((w // 2 - 22, h - 24), "Round", fill=(0, 0, 0), **kw)
    draw.text((8, (top + bottom) // 2 - 12), "Rate", fill=(0, 0, 0), **kw)
    img.save(path, "PNG")


def plot_clusters_all_vs_visible_pil(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    if "clusters_all" not in metrics_df.columns or "clusters_visible" not in metrics_df.columns:
        return
    w, h = 700, 420
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    r = pd.to_numeric(metrics_df["round"], errors="coerce").to_numpy(dtype=np.float64)
    a = pd.to_numeric(metrics_df["clusters_all"], errors="coerce").to_numpy(dtype=np.float64)
    v = pd.to_numeric(metrics_df["clusters_visible"], errors="coerce").to_numpy(dtype=np.float64)
    left, right, top, bottom = 56, w - 40, 48, h - 52
    draw.rectangle([left, top, right, bottom], outline=(180, 180, 180), width=1)
    mask = np.isfinite(r) & np.isfinite(a) & np.isfinite(v)
    if mask.sum() >= 2:
        rr, aa, vv = r[mask], a[mask], v[mask]
        xmin, xmax = rr.min(), rr.max()
        ymin, ymax = np.nanmin(np.concatenate([aa, vv])), np.nanmax(np.concatenate([aa, vv]))
        if xmax <= xmin:
            xmax = xmin + 1
        if ymax <= ymin:
            ymax = ymin + 1
        xp = left + (rr - xmin) / (xmax - xmin) * (right - left - 4)
        ya = bottom - (aa - ymin) / (ymax - ymin) * (bottom - top - 4)
        yv = bottom - (vv - ymin) / (ymax - ymin) * (bottom - top - 4)
        for i in range(len(rr) - 1):
            draw.line([(xp[i], ya[i]), (xp[i + 1], ya[i + 1])], fill=(31, 119, 180), width=2)
            draw.line([(xp[i], yv[i]), (xp[i + 1], yv[i + 1])], fill=(255, 127, 14), width=2)
    draw.text((16, 12), "Clusters: all vs visible", fill=(0, 0, 0), **kw)
    draw.text((w // 2 - 22, h - 24), "Round", fill=(0, 0, 0), **kw)
    draw.line([(left, 28), (left + 24, 28)], fill=(31, 119, 180), width=3)
    draw.text((left + 30, 22), "clusters_all", fill=(0, 0, 0), **kw)
    draw.line([(left + 150, 28), (left + 174, 28)], fill=(255, 127, 14), width=3)
    draw.text((left + 180, 22), "clusters_visible", fill=(0, 0, 0), **kw)
    img.save(path, "PNG")


def plot_topk_turnover_pil(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    if "topk_turnover_rate" not in metrics_df.columns:
        return
    w, h = 700, 420
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    r = pd.to_numeric(metrics_df["round"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(metrics_df["topk_turnover_rate"], errors="coerce").to_numpy(dtype=np.float64)
    left, right, top, bottom = 56, w - 40, 48, h - 52
    draw.rectangle([left, top, right, bottom], outline=(180, 180, 180), width=1)
    mask = np.isfinite(r) & np.isfinite(y)
    if mask.sum() >= 2:
        rr, yy = r[mask], np.clip(y[mask], 0.0, 1.0)
        xmin, xmax = rr.min(), rr.max()
        if xmax <= xmin:
            xmax = xmin + 1
        xp = left + (rr - xmin) / (xmax - xmin) * (right - left - 4)
        yp = bottom - yy * (bottom - top - 4)
        for i in range(len(rr) - 1):
            draw.line([(xp[i], yp[i]), (xp[i + 1], yp[i + 1])], fill=(140, 86, 75), width=2)
    draw.text((16, 12), "Top-K turnover rate", fill=(0, 0, 0), **kw)
    draw.text((w // 2 - 22, h - 24), "Round", fill=(0, 0, 0), **kw)
    draw.text((8, (top + bottom) // 2 - 10), "Turnover", fill=(0, 0, 0), **kw)
    img.save(path, "PNG")


def plot_institution_vs_generation_pil(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    needed = ["generated_count", "rgb_std_mean", "center_deviation_proxy"]
    if not all(c in metrics_df.columns for c in needed):
        return
    w, h = 760, 900
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    r = pd.to_numeric(metrics_df["round"], errors="coerce").to_numpy(dtype=np.float64)
    panels = [
        ("Generation activity", [("generated_count", (31, 119, 180))]),
        ("Generation style proxies", [("rgb_std_mean", (44, 160, 44)), ("center_deviation_proxy", (148, 103, 189))]),
        ("Institution metric", [("modularity", (214, 39, 40)) if "modularity" in metrics_df.columns else ("center_deviation_proxy", (148, 103, 189))]),
    ]
    panel_h = 260
    for p_idx, (title, series) in enumerate(panels):
        x0, y0 = 60, 40 + p_idx * (panel_h + 20)
        x1, y1 = w - 40, y0 + panel_h
        draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)
        vals = []
        for col, _ in series:
            vals.append(pd.to_numeric(metrics_df[col], errors="coerce").to_numpy(dtype=np.float64))
        yall = np.concatenate([v[np.isfinite(v)] for v in vals if np.isfinite(v).any()]) if vals else np.array([])
        if yall.size == 0:
            draw.text((x0 + 8, y0 + 6), f"{title} (no data)", fill=(0, 0, 0), **kw)
            continue
        xmin, xmax = np.nanmin(r), np.nanmax(r)
        ymin, ymax = np.nanmin(yall), np.nanmax(yall)
        if xmax <= xmin:
            xmax = xmin + 1
        if ymax <= ymin:
            ymax = ymin + 1
        draw.text((x0 + 8, y0 + 6), title, fill=(0, 0, 0), **kw)
        for col, color in series:
            v = pd.to_numeric(metrics_df[col], errors="coerce").to_numpy(dtype=np.float64)
            mask = np.isfinite(r) & np.isfinite(v)
            if mask.sum() < 2:
                continue
            rr, vv = r[mask], v[mask]
            xp = x0 + 30 + (rr - xmin) / (xmax - xmin) * (x1 - x0 - 50)
            yp = y1 - 20 - (vv - ymin) / (ymax - ymin) * (y1 - y0 - 40)
            for i in range(len(xp) - 1):
                draw.line([(xp[i], yp[i]), (xp[i + 1], yp[i + 1])], fill=color, width=2)
    img.save(path, "PNG")


def plot_vlm_dynamics_pil(metrics_df: pd.DataFrame, path: Path) -> None:
    if Image is None or ImageDraw is None:
        return
    cols = [
        "vlm_style_conformity_mean",
        "vlm_novelty_score_mean",
        "vlm_prompt_alignment_mean",
        "vlm_craft_score_mean",
    ]
    if not all(c in metrics_df.columns for c in cols):
        return
    w, h = 760, 460
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(12)
    kw = {} if font is None else {"font": font}
    r = pd.to_numeric(metrics_df["round"], errors="coerce").to_numpy(dtype=np.float64)
    vals = [pd.to_numeric(metrics_df[c], errors="coerce").to_numpy(dtype=np.float64) for c in cols]
    left, right, top, bottom = 60, w - 40, 44, h - 52
    draw.rectangle([left, top, right, bottom], outline=(180, 180, 180), width=1)
    finite_r = r[np.isfinite(r)]
    if finite_r.size == 0:
        img.save(path, "PNG")
        return
    xmin, xmax = float(np.nanmin(finite_r)), float(np.nanmax(finite_r))
    if xmax <= xmin:
        xmax = xmin + 1
    ymin, ymax = 0.0, 1.0
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189)]
    for vi, v in enumerate(vals):
        mask = np.isfinite(r) & np.isfinite(v)
        if mask.sum() < 2:
            continue
        rr, yy = r[mask], np.clip(v[mask], 0.0, 1.0)
        xp = left + (rr - xmin) / (xmax - xmin) * (right - left - 4)
        yp = bottom - (yy - ymin) / (ymax - ymin) * (bottom - top - 4)
        for i in range(len(rr) - 1):
            draw.line([(xp[i], yp[i]), (xp[i + 1], yp[i + 1])], fill=colors[vi], width=2)
    draw.text((16, 12), "VLM evaluation dynamics", fill=(0, 0, 0), **kw)
    legend = ["style_conformity", "novelty_score", "prompt_alignment", "craft_score"]
    lx = left
    for i, name in enumerate(legend):
        draw.line([(lx, 28), (lx + 22, 28)], fill=colors[i], width=3)
        draw.text((lx + 26, 22), name, fill=(0, 0, 0), **kw)
        lx += 170
    draw.text((w // 2 - 22, h - 24), "Round", fill=(0, 0, 0), **kw)
    draw.text((8, (top + bottom) // 2 - 10), "Score", fill=(0, 0, 0), **kw)
    img.save(path, "PNG")


def run_all(
    output_dir: Path,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    influence: Optional[np.ndarray] = None,
    metrics_df: Optional[pd.DataFrame] = None,
) -> None:
    import gc
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    coords = np.ascontiguousarray(np.asarray(coords_2d, dtype=np.float64))
    lbls = np.ascontiguousarray(np.asarray(labels, dtype=np.int32))
    inf_arr = np.ascontiguousarray(np.asarray(influence, dtype=np.float64)) if influence is not None else None
    plot_umap_cluster(coords, lbls, plot_dir / "umap_cluster.png")
    gc.collect()
    if inf_arr is not None:
        plot_umap_influence(coords, inf_arr, plot_dir / "umap_influence.png")
        gc.collect()
    if metrics_df is not None and not metrics_df.empty:
        plot_metrics_curves(metrics_df.copy(), plot_dir / "metrics_curves.png")
        gc.collect()
        plot_intra_inter(metrics_df.copy(), plot_dir / "intra_inter_dist.png")
        gc.collect()
        plot_separation_ratio(metrics_df.copy(), plot_dir / "separation_ratio.png")
        gc.collect()
        plot_game_dynamics_pil(metrics_df.copy(), plot_dir / "game_dynamics.png")
        gc.collect()
        plot_cross_cluster_rate_pil(metrics_df.copy(), plot_dir / "citation_cross_rate.png")
        gc.collect()
        plot_clusters_all_vs_visible_pil(metrics_df.copy(), plot_dir / "clusters_all_vs_visible.png")
        gc.collect()
        plot_topk_turnover_pil(metrics_df.copy(), plot_dir / "topk_turnover_rate.png")
        gc.collect()
        plot_institution_vs_generation_pil(metrics_df.copy(), plot_dir / "institution_vs_generation.png")
        gc.collect()
        plot_vlm_dynamics_pil(metrics_df.copy(), plot_dir / "vlm_dynamics.png")
        gc.collect()


def main_cli() -> None:
    """Entry point for subprocess: load from output_dir and run_all. Usage: python -m src.viz <output_dir>"""
    import sys
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description="Render plots from output directory")
    parser.add_argument("output_dir", type=str, help="Directory containing metrics and arrays")
    parser.add_argument("--engine", choices=["auto", "matplotlib", "pil"], default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    umap_path = output_dir / "umap_2d.npy"
    labels_path = output_dir / "cluster_labels.npy"
    nodes_path = output_dir / "nodes.csv"
    missing = [p for p in (umap_path, labels_path, nodes_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required plotting inputs: " + ", ".join(str(p) for p in missing))
    coords_2d = np.load(umap_path)
    labels = np.load(labels_path)
    nodes = pd.read_csv(nodes_path)
    influence_col = None
    for c in ("I", "influence", "pagerank", "centrality", "I_score"):
        if c in nodes.columns:
            influence_col = c
            break
    if influence_col is None:
        raise ValueError(
            f"{nodes_path} missing influence column. Expected one of: "
            "I, influence, pagerank, centrality, I_score"
        )
    if influence_col != "I":
        print(f"[viz][WARNING] Using fallback influence column '{influence_col}' from nodes.csv")
    influence = pd.to_numeric(nodes[influence_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    metrics_path = output_dir / "metrics.csv"
    metrics_df = pd.read_csv(metrics_path) if metrics_path.is_file() else None
    if metrics_df is not None:
        if "round" in metrics_df.columns:
            metrics_df["round"] = pd.to_numeric(metrics_df["round"], errors="coerce")
            if metrics_df["round"].isna().all():
                print("[viz][WARNING] metrics.csv has round column but all values are NaN; skip metrics plots.")
                metrics_df = None
        else:
            print("[viz][WARNING] metrics.csv missing round column; skip metrics plots.")
            metrics_df = None
    engine = args.engine
    if engine == "auto":
        engine = "matplotlib" if _MPL_OK else "pil"
    if engine == "matplotlib":
        print("[viz] rendering with matplotlib")
        run_all_matplotlib(output_dir, coords_2d, labels, influence=influence, metrics_df=metrics_df)
    else:
        print("[viz] rendering with PIL")
        run_all(output_dir, coords_2d, labels, influence=influence, metrics_df=metrics_df)


if __name__ == "__main__":
    main_cli()
