from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# Headless plotting (prevents Tkinter crashes)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.exp_config import get_paths
from experiments.exp_utils import ensure_dir, read_json

BG_COLOR = "#0B0B10"
FG_COLOR = "#FFFFFF"
ACCENT = "#E10600"
GRID_ALPHA = 0.12
BAR_COLOR = "#2B2B35"
FOOTER_TEXT = "Leakage-safe GroupKFold (by race) - FYP Pit Stop Prediction"


def apply_f1_style(ax, fig, title: str | None = None, subtitle: str | None = None, x_rotation: int = 0) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )

    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=FG_COLOR, alpha=GRID_ALPHA, linewidth=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8A8A8")
    ax.spines["bottom"].set_color("#A8A8A8")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    ax.tick_params(axis="both", colors=FG_COLOR, labelsize=10)
    ax.xaxis.label.set_color(FG_COLOR)
    ax.yaxis.label.set_color(FG_COLOR)

    for label in ax.get_xticklabels():
        label.set_rotation(x_rotation)
        label.set_ha("right" if x_rotation else "center")

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", color=ACCENT, pad=14)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=FG_COLOR,
            fontsize=11,
        )

    fig.text(
        0.99,
        0.01,
        FOOTER_TEXT,
        ha="right",
        va="bottom",
        color=FG_COLOR,
        alpha=0.7,
        fontsize=9,
    )


def _mean_std(metric: dict, key: str) -> tuple[float, float]:
    vals = [f[key] for f in metric["folds"]]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _load_all(paths):
    m1 = read_json(paths.out_replication / "metrics.json")
    m2 = read_json(paths.out_refdata_mymethod / "metrics.json")
    m3 = read_json(paths.out_mydata_reftech_weather / "metrics.json")
    m4 = read_json(paths.out_mydata_mymethod_weather / "metrics.json")
    return m1, m2, m3, m4


def _stage_colors() -> list[str]:
    return [BAR_COLOR, ACCENT, BAR_COLOR, ACCENT]


def plot_stage_bars(m1, m2, m3, m4, outdir: Path, strict_tag: str = "STRICT GroupKFold") -> None:
    labels = [
        "Stage 1: RefTech on RefData",
        "Stage 2: MyMethod on RefData",
        "Stage 3: RefTech on MyData+W",
        "Stage 4: MyMethod on MyData+W",
    ]
    means = [m1["mean_f1"], m2["mean_f1"], m3["mean_f1"], m4["mean_f1"]]
    stds = [m1["std_f1"], m2["std_f1"], m3["std_f1"], m4["std_f1"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color=_stage_colors(), edgecolor="#3A3A48")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean F1-score (Pit-stop class = 1)")
    apply_f1_style(
        ax,
        fig,
        title=f"Supervisor Workflow Comparison (Stage 1 to Stage 4) - {strict_tag}",
        x_rotation=20,
    )
    for i, v in enumerate(means):
        ax.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", color=FG_COLOR)
    fig.tight_layout()
    fig.savefig(outdir / "f1_mean_std_stages.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_recall_bars(m1, m2, m3, m4, outdir: Path) -> None:
    labels = [
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
    ]
    r_means = []
    r_stds = []
    for m in [m1, m2, m3, m4]:
        mu, sd = _mean_std(m, "recall")
        r_means.append(mu)
        r_stds.append(sd)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    ax.bar(x, r_means, yerr=r_stds, capsize=6, color=_stage_colors(), edgecolor="#3A3A48")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean Recall (Pit-stop class = 1)")
    apply_f1_style(
        ax,
        fig,
        title="Pit-stop Detection Recall (Stage 1 to Stage 4) - Unseen Races (GroupKFold)",
        x_rotation=0,
    )
    for i, v in enumerate(r_means):
        ax.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", color=FG_COLOR)
    fig.tight_layout()
    fig.savefig(outdir / "recall_mean_std_stages.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_pr_auc_bars(m1, m2, m3, m4, outdir: Path) -> None:
    labels = [
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
    ]
    p_means = []
    p_stds = []
    for m in [m1, m2, m3, m4]:
        mu, sd = _mean_std(m, "pr_auc")
        p_means.append(mu)
        p_stds.append(sd)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    ax.bar(x, p_means, yerr=p_stds, capsize=6, color=_stage_colors(), edgecolor="#3A3A48")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean PR-AUC (Pit-stop class = 1)")
    apply_f1_style(ax, fig, title="PR-AUC by Stage - Unseen Races (GroupKFold)", x_rotation=0)
    for i, v in enumerate(p_means):
        ax.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", color=FG_COLOR)
    fig.tight_layout()
    fig.savefig(outdir / "pr_auc_mean_std_stages.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_delta(title: str, deltas: list[float], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, len(deltas) + 1)
    ax.bar(x, deltas, color=BAR_COLOR, edgecolor="#3A3A48")
    ax.axhline(0.0, linewidth=0.8, color=FG_COLOR, alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x])
    ax.set_ylabel("Delta F1 (Pit-stop class)")
    mu = float(np.mean(deltas))
    ax.axhline(mu, linewidth=1.0, color=ACCENT, alpha=0.9)
    apply_f1_style(ax, fig, title=f"{title}\nMean Delta F1 = {mu:+.3f}", x_rotation=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary_csv(m1, m2, m3, m4, outpath: Path) -> None:
    rows = []
    stages = [
        ("Stage 1: RefTech on RefData", m1),
        ("Stage 2: MyMethod on RefData", m2),
        ("Stage 3: RefTech on MyData+W", m3),
        ("Stage 4: MyMethod on MyData+W", m4),
    ]
    for name, m in stages:
        prec_mu, prec_sd = _mean_std(m, "precision")
        rec_mu, rec_sd = _mean_std(m, "recall")
        pr_mu, pr_sd = _mean_std(m, "pr_auc")
        f1_mu, f1_sd = m["mean_f1"], m["std_f1"]
        rows.append(
            dict(
                stage=name,
                mean_f1=f1_mu,
                std_f1=f1_sd,
                mean_precision=prec_mu,
                std_precision=prec_sd,
                mean_recall=rec_mu,
                std_recall=rec_sd,
                mean_pr_auc=pr_mu,
                std_pr_auc=pr_sd,
                threshold=m.get("threshold", 0.5),
                n_splits=m.get("n_splits", 5),
            )
        )
    pd.DataFrame(rows).to_csv(outpath, index=False)


def main() -> None:
    paths = get_paths()
    ensure_dir(paths.out_summary_plots)

    m1, m2, m3, m4 = _load_all(paths)

    # Main stage comparison
    plot_stage_bars(m1, m2, m3, m4, paths.out_summary_plots)

    # Recall (important for pit-stop detection)
    plot_recall_bars(m1, m2, m3, m4, paths.out_summary_plots)

    # PR-AUC (imbalance-aware)
    plot_pr_auc_bars(m1, m2, m3, m4, paths.out_summary_plots)

    # Improvement consistency plots
    d21 = [f2["f1"] - f1["f1"] for f1, f2 in zip(m1["folds"], m2["folds"])]
    d43 = [f4["f1"] - f3["f1"] for f3, f4 in zip(m3["folds"], m4["folds"])]

    plot_delta(
        "Improvement Consistency on Reference Dataset\nStage 2 vs Stage 1 (Delta = MyMethod - RefTech)",
        d21,
        paths.out_summary_plots / "delta_f1_stage2_minus_stage1.png",
    )
    plot_delta(
        "Improvement Consistency on Personal Dataset + Weather\nStage 4 vs Stage 3 (Delta = MyMethod - RefTech)",
        d43,
        paths.out_summary_plots / "delta_f1_stage4_minus_stage3.png",
    )

    # Summary table
    write_summary_csv(m1, m2, m3, m4, paths.out_summary_plots / "stage_summary_strict.csv")

    print("[OK] Saved plots + CSV summary to:", paths.out_summary_plots)


if __name__ == "__main__":
    # run from project root:
    #   python -m experiments.exp_plot_all
    main()
