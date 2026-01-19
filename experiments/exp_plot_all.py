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


def _load_all_standard(paths):
    m1 = read_json(paths.out_replication / "metrics.json")
    m2 = read_json(paths.out_refdata_mymethod / "metrics.json")
    std_ref = paths.results_dir / "standard_mydata_reftech" / "metrics.json"
    std_my = paths.results_dir / "standard_mydata_mymethod" / "metrics.json"
    if not std_ref.exists() or not std_my.exists():
        return None
    m3 = read_json(std_ref)
    m4 = read_json(std_my)
    return m1, m2, m3, m4


def _stage_colors() -> list[str]:
    return [BAR_COLOR, ACCENT, BAR_COLOR, ACCENT]


def _format_fbeta_label(beta: float) -> str:
    if abs(beta - round(beta)) < 1e-6:
        return f"F{int(round(beta))}"
    return f"Fbeta (beta={beta:g})"


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


def plot_fbeta_bars(m1, m2, m3, m4, outdir: Path, beta: float) -> None:
    labels = [
        "Stage 1: RefTech on RefData",
        "Stage 2: MyMethod on RefData",
        "Stage 3: RefTech on MyData+W",
        "Stage 4: MyMethod on MyData+W",
    ]
    means = [m1["mean_fbeta"], m2["mean_fbeta"], m3["mean_fbeta"], m4["mean_fbeta"]]
    stds = [m1["std_fbeta"], m2["std_fbeta"], m3["std_fbeta"], m4["std_fbeta"]]
    metric_label = _format_fbeta_label(beta)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color=_stage_colors(), edgecolor="#3A3A48")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"Mean {metric_label}-score (Pit-stop class = 1)")
    apply_f1_style(
        ax,
        fig,
        title=f"{metric_label} by Stage - Unseen Races (GroupKFold)",
        x_rotation=20,
    )
    for i, v in enumerate(means):
        ax.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", color=FG_COLOR)
    fig.tight_layout()
    fig.savefig(outdir / "fbeta_mean_std_stages.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
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
        fbeta_mu = m.get("mean_fbeta")
        fbeta_sd = m.get("std_fbeta")
        if fbeta_mu is None and m.get("folds") and "fbeta" in m["folds"][0]:
            fbeta_mu, fbeta_sd = _mean_std(m, "fbeta")
        threshold_mean = m.get("mean_threshold", m.get("threshold", 0.5))
        threshold_std = m.get("std_threshold", 0.0)
        beta = m.get("beta", 1.0)
        rows.append(
            dict(
                stage=name,
                mean_f1=f1_mu,
                std_f1=f1_sd,
                mean_fbeta=fbeta_mu,
                std_fbeta=fbeta_sd,
                mean_precision=prec_mu,
                std_precision=prec_sd,
                mean_recall=rec_mu,
                std_recall=rec_sd,
                mean_pr_auc=pr_mu,
                std_pr_auc=pr_sd,
                threshold=threshold_mean,
                std_threshold=threshold_std,
                n_splits=m.get("n_splits", 5),
                beta=beta,
            )
        )
    pd.DataFrame(rows).to_csv(outpath, index=False)


def write_summary_csv_basic(m1, m2, m3, m4, outpath: Path, evaluation: str | None = None) -> None:
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
        f1_mu, f1_sd = m["mean_f1"], m["std_f1"]
        folds = m.get("folds", [])
        total_pos = sum(f.get("support_pos", 0) for f in folds) if folds else 0
        total_neg = sum(f.get("support_neg", 0) for f in folds) if folds else 0
        rows.append(
            dict(
                stage=name,
                mean_f1=f1_mu,
                std_f1=f1_sd,
                mean_precision=prec_mu,
                std_precision=prec_sd,
                mean_recall=rec_mu,
                std_recall=rec_sd,
                total_pos=int(total_pos),
                total_neg=int(total_neg),
                n_folds=m.get("n_splits", 5),
                evaluation=evaluation if evaluation else "",
            )
        )
    pd.DataFrame(rows).to_csv(outpath, index=False)


def write_fold_csv(m1, m2, m3, m4, outpath: Path) -> None:
    rows = []
    stages = [
        ("Stage 1: RefTech on RefData", m1),
        ("Stage 2: MyMethod on RefData", m2),
        ("Stage 3: RefTech on MyData+W", m3),
        ("Stage 4: MyMethod on MyData+W", m4),
    ]
    for name, m in stages:
        beta = m.get("beta", 1.0)
        for f in m.get("folds", []):
            rows.append(
                dict(
                    stage=name,
                    fold=f.get("fold"),
                    f1=f.get("f1"),
                    fbeta=f.get("fbeta"),
                    precision=f.get("precision"),
                    recall=f.get("recall"),
                    pr_auc=f.get("pr_auc"),
                    threshold=f.get("threshold"),
                    beta=beta,
                )
            )
    pd.DataFrame(rows).to_csv(outpath, index=False)


def main() -> None:
    paths = get_paths()
    ensure_dir(paths.out_summary_plots)

    m1, m2, m3, m4 = _load_all(paths)

    # Main stage comparison
    plot_stage_bars(m1, m2, m3, m4, paths.out_summary_plots)

    if "mean_fbeta" in m1:
        beta = float(m1.get("beta", 1.0))
        plot_fbeta_bars(m1, m2, m3, m4, paths.out_summary_plots, beta)

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

    if m1.get("folds") and "fbeta" in m1["folds"][0]:
        beta = float(m1.get("beta", 1.0))
        metric_label = _format_fbeta_label(beta)
        d21b = [f2["fbeta"] - f1["fbeta"] for f1, f2 in zip(m1["folds"], m2["folds"])]
        d43b = [f4["fbeta"] - f3["fbeta"] for f3, f4 in zip(m3["folds"], m4["folds"])]
        plot_delta(
            f"Improvement Consistency on Reference Dataset\nStage 2 vs Stage 1 (Delta {metric_label} = MyMethod - RefTech)",
            d21b,
            paths.out_summary_plots / "delta_fbeta_stage2_minus_stage1.png",
        )
        plot_delta(
            f"Improvement Consistency on Personal Dataset + Weather\nStage 4 vs Stage 3 (Delta {metric_label} = MyMethod - RefTech)",
            d43b,
            paths.out_summary_plots / "delta_fbeta_stage4_minus_stage3.png",
        )

    # Summary table (strict)
    write_summary_csv(m1, m2, m3, m4, paths.out_summary_plots / "stage_summary_strict.csv")
    write_fold_csv(m1, m2, m3, m4, paths.out_summary_plots / "stage_folds_strict.csv")

    standard = _load_all_standard(paths)
    if standard is not None:
        s1, s2, s3, s4 = standard
        write_summary_csv_basic(
            s1,
            s2,
            s3,
            s4,
            paths.out_summary_plots / "stage_summary.csv",
            evaluation="standard",
        )
        write_fold_csv(s1, s2, s3, s4, paths.out_summary_plots / "stage_folds_standard.csv")
    else:
        write_summary_csv_basic(
            m1,
            m2,
            m3,
            m4,
            paths.out_summary_plots / "stage_summary.csv",
            evaluation="strict",
        )

    print("[OK] Saved plots + CSV summary to:", paths.out_summary_plots)


if __name__ == "__main__":
    # run from project root:
    #   python -m experiments.exp_plot_all
    main()
