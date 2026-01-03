from __future__ import annotations

import argparse
import sys
import textwrap
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BG_COLOR = "#0B0F14"
FG_COLOR = "#FFFFFF"
ACCENT = "#E10600"
BAR_COLOR = "#2B2B35"
GRID_ALPHA = 0.08
FOOTER_TEXT = "Leakage-safe GroupKFold (by race) - FYP Pit Stop Prediction"

STAGE_ORDER = [
    "Stage 1: RefTech on RefData",
    "Stage 2: MyMethod on RefData",
    "Stage 3: RefTech on MyData+W",
    "Stage 4: MyMethod on MyData+W",
]

STAGE_SUBS = {
    1: "RefData",
    2: "RefData",
    3: "MyData+W",
    4: "MyData+W",
}

METRIC_ALIASES = {
    "f1": ["mean_f1", "f1_mean", "f1", "f1score", "mean_f1score", "f1_score"],
    "precision": ["mean_precision", "precision_mean", "precision", "prec", "mean_prec"],
    "recall": ["mean_recall", "recall_mean", "recall", "tpr", "sensitivity"],
    "pr_auc": ["mean_pr_auc", "pr_auc_mean", "pr_auc", "prauc", "average_precision"],
}

STD_ALIASES = {
    "f1": ["std_f1", "f1_std", "std_f1score", "f1_std_dev"],
    "precision": ["std_precision", "precision_std", "std_prec"],
    "recall": ["std_recall", "recall_std", "std_tpr"],
    "pr_auc": ["std_pr_auc", "pr_auc_std", "std_prauc"],
}

RAW_ALIASES = {
    "f1": ["f1", "f1_score", "f1score"],
    "precision": ["precision", "prec"],
    "recall": ["recall", "tpr"],
    "pr_auc": ["pr_auc", "prauc", "average_precision"],
}


def _normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    norm_map = {_normalize(c): c for c in df.columns}
    for c in candidates:
        key = _normalize(c)
        if key in norm_map:
            return norm_map[key]
    return None


def _detect_stage_col(df: pd.DataFrame) -> str:
    stage_candidates = ["stage", "Stage", "stage_name", "stageName", "experiment", "stage_label"]
    col = _find_col(df, stage_candidates)
    if col is not None:
        return col

    # fallback: pick a single object column if available
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if len(obj_cols) == 1:
        return obj_cols[0]

    cols_preview = ", ".join(df.columns.tolist())
    raise ValueError(
        "Cannot find stage label column. Expected one of: "
        f"{', '.join(stage_candidates)}.\nAvailable columns: {cols_preview}"
    )


def _wrap_labels(labels: List[str], width: int = 18) -> List[str]:
    return [textwrap.fill(lbl, width=width) for lbl in labels]


def _stage_tick_labels(labels: List[str]) -> List[str]:
    out = []
    for lbl in labels:
        rank = _stage_rank(str(lbl))
        if rank in STAGE_SUBS:
            out.append(f"S{rank}\n{STAGE_SUBS[rank]}")
        else:
            out.append(str(lbl))
    return out


def apply_f1_style(
    ax,
    fig,
    title: str | None = None,
    subtitle: str | None = None,
    title_size: int = 18,
    label_size: int = 14,
    tick_size: int = 12,
    x_rotation: int = 0,
    footer_note: str | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
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
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="both", colors=FG_COLOR, labelsize=tick_size)
    ax.tick_params(axis="x", pad=8)
    ax.xaxis.label.set_color(FG_COLOR)
    ax.yaxis.label.set_color(FG_COLOR)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)

    for label in ax.get_xticklabels():
        label.set_rotation(x_rotation)
        label.set_ha("right" if x_rotation else "center")

    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", color=ACCENT, pad=16)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=FG_COLOR,
            fontsize=12,
        )

    footer = FOOTER_TEXT
    if footer_note:
        footer = f"{FOOTER_TEXT} - {footer_note}"

    fig.text(
        0.99,
        0.01,
        footer,
        ha="right",
        va="bottom",
        color=FG_COLOR,
        alpha=0.7,
        fontsize=9,
    )


def _stage_rank(label: str) -> int:
    low = label.lower()
    for i in range(1, 5):
        if f"stage {i}" in low:
            return i
    return 99


def _order_stages(df: pd.DataFrame, stage_col: str) -> pd.DataFrame:
    out = df.copy()
    out["__stage_rank"] = out[stage_col].apply(_stage_rank)
    out = out.sort_values(by="__stage_rank")
    return out.drop(columns="__stage_rank")


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[ERROR] Input CSV not found: {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV: {csv_path}\n{exc}")
        sys.exit(1)
    if df.empty:
        print(f"[ERROR] CSV is empty: {csv_path}")
        sys.exit(1)
    return df


def compute_summary(df: pd.DataFrame, stage_col: str) -> Tuple[pd.DataFrame, Dict[str, bool], Dict[str, str]]:
    has_repeats = df[stage_col].duplicated().any()
    mean_cols = {m: _find_col(df, METRIC_ALIASES[m]) for m in METRIC_ALIASES}
    std_cols = {m: _find_col(df, STD_ALIASES[m]) for m in STD_ALIASES}
    raw_cols = {m: _find_col(df, RAW_ALIASES[m]) for m in RAW_ALIASES}

    std_missing = {m: False for m in METRIC_ALIASES}
    used_cols: Dict[str, str] = {}

    if has_repeats and any(raw_cols.values()):
        metrics_present = {m: c for m, c in raw_cols.items() if c is not None}
        grouped = df.groupby(stage_col, dropna=False)
        rows = []
        for stage, sub in grouped:
            row = {"stage": stage}
            for metric, col in metrics_present.items():
                vals = sub[col].astype(float)
                row[f"mean_{metric}"] = float(vals.mean())
                row[f"std_{metric}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            rows.append(row)
        summary = pd.DataFrame(rows)
        used_cols = {m: raw_cols[m] for m in metrics_present}
    elif any(mean_cols.values()):
        rows = []
        for _, r in df.iterrows():
            row = {"stage": r[stage_col]}
            for metric, col in mean_cols.items():
                if col is None:
                    continue
                row[f"mean_{metric}"] = float(r[col])
                std_col = std_cols.get(metric)
                if std_col is None:
                    row[f"std_{metric}"] = 0.0
                    std_missing[metric] = True
                else:
                    row[f"std_{metric}"] = float(r[std_col])
            rows.append(row)
        summary = pd.DataFrame(rows)
        used_cols = {m: mean_cols[m] for m in mean_cols if mean_cols[m] is not None}
    elif any(raw_cols.values()):
        rows = []
        for _, r in df.iterrows():
            row = {"stage": r[stage_col]}
            for metric, col in raw_cols.items():
                if col is None:
                    continue
                row[f"mean_{metric}"] = float(r[col])
                row[f"std_{metric}"] = 0.0
                std_missing[metric] = True
            rows.append(row)
        summary = pd.DataFrame(rows)
        used_cols = {m: raw_cols[m] for m in raw_cols if raw_cols[m] is not None}
    else:
        cols_preview = ", ".join(df.columns.tolist())
        raise ValueError(
            "No usable metric columns found. "
            "Expected mean/metric columns like mean_f1, precision, recall, pr_auc.\n"
            f"Available columns: {cols_preview}"
        )

    summary = _order_stages(summary, "stage")
    return summary, std_missing, used_cols


def _stage_colors(n: int) -> List[str]:
    colors = [BAR_COLOR] * n
    for idx in (1, 3):
        if idx < n:
            colors[idx] = ACCENT
    return colors


def _metric_arrays(summary: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    mean_col = f"mean_{metric}"
    std_col = f"std_{metric}"
    if mean_col not in summary.columns:
        return np.array([]), np.array([]), []
    means = summary[mean_col].values.astype(float)
    stds = summary[std_col].values.astype(float) if std_col in summary.columns else np.zeros_like(means)
    labels = summary["stage"].astype(str).tolist()
    return means, stds, labels


def _stage_value(summary: pd.DataFrame, stage_num: int, metric: str) -> float | None:
    mean_col = f"mean_{metric}"
    if mean_col not in summary.columns:
        return None
    for _, row in summary.iterrows():
        if _stage_rank(row["stage"]) == stage_num:
            return float(row[mean_col])
    return None


def _extract_fold_deltas(df: pd.DataFrame, stage_col: str, metric: str) -> Tuple[List[float], List[float]]:
    fold_cols = []
    for c in df.columns:
        m = re.match(rf"{metric}_fold(\d+)$", _normalize(c))
        if m:
            fold_cols.append((int(m.group(1)), c))

    if fold_cols:
        fold_cols = sorted(fold_cols, key=lambda x: x[0])
        stage_map = {row[stage_col]: row for _, row in df.iterrows()}
        def _stage_row(stage_num: int):
            for s in stage_map:
                if _stage_rank(str(s)) == stage_num:
                    return stage_map[s]
            return None
        s1 = _stage_row(1)
        s2 = _stage_row(2)
        s3 = _stage_row(3)
        s4 = _stage_row(4)
        if s1 is not None and s2 is not None:
            d21 = [float(s2[c]) - float(s1[c]) for _, c in fold_cols]
        else:
            d21 = []
        if s3 is not None and s4 is not None:
            d43 = [float(s4[c]) - float(s3[c]) for _, c in fold_cols]
        else:
            d43 = []
        return d21, d43

    fold_col = _find_col(df, ["fold", "fold_id", "cv_fold"])
    metric_col = _find_col(df, RAW_ALIASES[metric])
    if fold_col and metric_col:
        pivot = df.pivot_table(index=fold_col, columns=stage_col, values=metric_col, aggfunc="mean")
        d21 = []
        d43 = []
        for fold in sorted(pivot.index.tolist()):
            row = pivot.loc[fold]
            s1 = None
            s2 = None
            s3 = None
            s4 = None
            for stage in row.index:
                rank = _stage_rank(str(stage))
                if rank == 1:
                    s1 = row[stage]
                elif rank == 2:
                    s2 = row[stage]
                elif rank == 3:
                    s3 = row[stage]
                elif rank == 4:
                    s4 = row[stage]
            if s1 is not None and s2 is not None:
                d21.append(float(s2 - s1))
            if s3 is not None and s4 is not None:
                d43.append(float(s4 - s3))
        return d21, d43

    return [], []


def plot_poster_main(summary: pd.DataFrame, outpath: Path, std_missing: bool) -> Tuple[float, float]:
    means, stds, labels = _metric_arrays(summary, "f1")
    if means.size == 0:
        raise ValueError("F1 metric is required for fig_poster_main.png")

    tick_labels = _stage_tick_labels(labels)
    fig, ax = plt.subplots(figsize=(13, 8))
    x = np.arange(len(tick_labels))
    ax.bar(x, means, yerr=stds, capsize=7, width=0.6, color=_stage_colors(len(tick_labels)), edgecolor="#3A3A48", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean F1-score (Pit-stop class = 1)")

    footer_note = "std unavailable" if std_missing else None
    apply_f1_style(
        ax,
        fig,
        title="LEAKAGE-FREE PIT-STOP PREDICTION RESULTS",
        subtitle="STRICT GroupKFold by Race - Mean +/- Std (Pit-stop class)",
        title_size=28,
        label_size=16,
        tick_size=13,
        x_rotation=0,
        footer_note=footer_note,
    )

    for i, v in enumerate(means):
        ax.text(i, min(1.02, v + 0.045), f"{v:.3f}", ha="center", va="bottom", color=FG_COLOR, fontsize=15)

    s1 = _stage_value(summary, 1, "f1")
    s2 = _stage_value(summary, 2, "f1")
    s3 = _stage_value(summary, 3, "f1")
    s4 = _stage_value(summary, 4, "f1")
    d21 = (s2 - s1) if s1 is not None and s2 is not None else float("nan")
    d43 = (s4 - s3) if s3 is not None and s4 is not None else float("nan")

    callout = f"Delta F1 Stage2-Stage1 = {d21:+.3f}\nDelta F1 Stage4-Stage3 = {d43:+.3f}"
    ax.text(
        0.98,
        0.92,
        callout,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=FG_COLOR,
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#14141A", edgecolor=ACCENT, linewidth=1.2),
    )

    fig.subplots_adjust(bottom=0.22)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return d21, d43


def _plot_metric_panel(ax, fig, summary: pd.DataFrame, metric: str, title: str, std_missing: bool, title_size: int = 18) -> None:
    means, stds, labels = _metric_arrays(summary, metric)
    if means.size == 0:
        ax.text(0.5, 0.5, f"{title}\nUnavailable", ha="center", va="center", color=FG_COLOR, fontsize=14)
        apply_f1_style(ax, fig, title=title, title_size=title_size)
        return

    tick_labels = _stage_tick_labels(labels)
    x = np.arange(len(tick_labels))
    ax.bar(x, means, yerr=stds, capsize=6, width=0.6, color=_stage_colors(len(tick_labels)), edgecolor="#3A3A48")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Mean {title}")

    footer_note = "std unavailable" if std_missing else None
    apply_f1_style(ax, fig, title=title, title_size=title_size, x_rotation=0, footer_note=footer_note)
    for i, v in enumerate(means):
        ax.text(i, min(1.0, v + 0.04), f"{v:.3f}", ha="center", va="bottom", color=FG_COLOR, fontsize=12)


def plot_slide_summary(summary: pd.DataFrame, outpath: Path, std_missing: Dict[str, bool]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    _plot_metric_panel(axes[0], fig, summary, "f1", "F1 (Pit-stop)", std_missing.get("f1", False), title_size=20)
    _plot_metric_panel(axes[1], fig, summary, "pr_auc", "PR-AUC", std_missing.get("pr_auc", False), title_size=20)
    fig.subplots_adjust(wspace=0.20, left=0.06, right=0.98, bottom=0.10, top=0.90)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_thesis_metrics(summary: pd.DataFrame, outpath: Path, std_missing: Dict[str, bool]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_metric_panel(axes[0, 0], fig, summary, "f1", "F1 (Pit-stop)", std_missing.get("f1", False))
    _plot_metric_panel(axes[0, 1], fig, summary, "precision", "Precision", std_missing.get("precision", False))
    _plot_metric_panel(axes[1, 0], fig, summary, "recall", "Recall", std_missing.get("recall", False))
    _plot_metric_panel(axes[1, 1], fig, summary, "pr_auc", "PR-AUC", std_missing.get("pr_auc", False))
    fig.subplots_adjust(wspace=0.25, hspace=0.28, left=0.06, right=0.98, bottom=0.08, top=0.92)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_delta_consistency(
    deltas: List[float],
    mean_delta: float | None,
    outpath: Path,
    subtitle: str,
    fallback_note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if deltas:
        x = np.arange(1, len(deltas) + 1)
        ax.bar(x, deltas, color=BAR_COLOR, edgecolor="#3A3A48", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i}" for i in x])
        mu = float(np.mean(deltas))
    else:
        x = np.arange(1, 2)
        value = mean_delta if mean_delta is not None else 0.0
        ax.bar(x, [value], color=BAR_COLOR, edgecolor="#3A3A48", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(["Mean"])
        mu = float(value)

    ax.axhline(0.0, linewidth=0.8, color=FG_COLOR, alpha=0.25)
    ax.axhline(mu, linewidth=1.2, color=ACCENT, alpha=0.9)
    ax.set_ylabel("Delta F1 (Pit-stop class)")
    ax.text(
        0.98,
        0.90,
        f"Mean Delta F1 = {mu:+.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=FG_COLOR,
        fontsize=12,
    )

    note = fallback_note if fallback_note else None
    apply_f1_style(
        ax,
        fig,
        title="IMPROVEMENT CONSISTENCY ACROSS UNSEEN RACES",
        subtitle=subtitle if not note else f"{subtitle} - {note}",
        title_size=18,
        label_size=14,
        tick_size=11,
        x_rotation=0,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F1 broadcast-style thesis plots.")
    parser.add_argument("--input", required=True, help="Path to stage_summary(_strict).csv")
    parser.add_argument("--outdir", required=True, help="Output directory for PNGs")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(input_path)
    stage_col = _detect_stage_col(df)
    summary, std_missing, used_cols = compute_summary(df, stage_col)

    missing_metrics = [m for m in METRIC_ALIASES if f"mean_{m}" not in summary.columns]
    if missing_metrics:
        print(f"[WARN] Missing metrics: {', '.join(missing_metrics)}")
    if "f1" in missing_metrics:
        print("[ERROR] F1 metric missing; cannot generate required poster plot.")
        sys.exit(1)

    d21, d43 = plot_poster_main(summary, outdir / "fig_poster_main.png", std_missing.get("f1", False))
    plot_slide_summary(summary, outdir / "fig_slide_summary.png", std_missing)
    plot_thesis_metrics(summary, outdir / "fig_thesis_metrics.png", std_missing)

    fold_d21, fold_d43 = _extract_fold_deltas(df, stage_col, "f1")
    fallback_note = None
    if not fold_d21:
        fallback_note = "mean-only (no fold detail available)"
    plot_delta_consistency(
        fold_d21,
        d21 if not np.isnan(d21) else None,
        outdir / "fig_delta_consistency_ref.png",
        subtitle="Stage 2 vs Stage 1 (RefData)",
        fallback_note=fallback_note,
    )

    fallback_note = None
    if not fold_d43:
        fallback_note = "mean-only (no fold detail available)"
    plot_delta_consistency(
        fold_d43,
        d43 if not np.isnan(d43) else None,
        outdir / "fig_delta_consistency_mydata.png",
        subtitle="Stage 4 vs Stage 3 (MyData+W)",
        fallback_note=fallback_note,
    )

    report_lines = [
        "[REPORT] Mean F1 per stage:",
    ]
    for _, row in summary.iterrows():
        if "mean_f1" in summary.columns:
            report_lines.append(f"  - {row['stage']}: {row['mean_f1']:.3f}")
    report_lines.append(f"[REPORT] Delta F1 Stage2-Stage1 = {d21:+.3f}")
    report_lines.append(f"[REPORT] Delta F1 Stage4-Stage3 = {d43:+.3f}")
    report_lines.append(f"[REPORT] Output dir: {outdir}")
    report_lines.append("[REPORT] Files:")
    for p in sorted(outdir.glob("*.png")):
        report_lines.append(f"  - {p.name}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
