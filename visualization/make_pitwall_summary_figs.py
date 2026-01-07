"""
Pit wall broadcast-style summary figures.

Inputs:
  --in_csv   Path to stage_summary.csv (and optional stage_summary_strict.csv)
Outputs:
  results/summary_plots/pitwall/
    - pitwall_figA_overview.png/.pdf
    - pitwall_figB_consistency.png/.pdf
    - pitwall_figC_slide.png/.pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


BG_COLOR = "#0B0F14"
PANEL_COLOR = "#12151C"
FG_COLOR = "#FFFFFF"
ACCENT = "#E10600"
REF_COLOR = "#8C9099"
GRID_ALPHA = 0.08
HEADER_H = 0.10
TITLE_SIZE = 28
SUBTITLE_SIZE = 13
AX_TITLE_SIZE = 18
TICK_SIZE = 13
VALUE_SIZE = 16
KPI_VALUE_SIZE = 26


def _normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    norm_map = {_normalize(c): c for c in df.columns}
    for c in candidates:
        key = _normalize(c)
        if key in norm_map:
            return norm_map[key]
    return None


def _stage_rank(value) -> int:
    if isinstance(value, (int, float)) and not pd.isna(value):
        n = int(value)
        return n if 1 <= n <= 4 else 99
    s = str(value)
    m = re.search(r"stage\s*([1-4])", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([1-4])\b", s)
    if m:
        return int(m.group(1))
    return 99


def _infer_method(label: str, stage_num: int | None) -> str:
    s = label.lower()
    if "mymethod" in s:
        return "MyMethod"
    if "reftech" in s or "ref" in s:
        return "RefTech"
    if stage_num in (2, 4):
        return "MyMethod"
    if stage_num in (1, 3):
        return "RefTech"
    return "RefTech"


def _format_title(text: str) -> str:
    text = text.upper()
    return "  ".join(text.split())


def _format_metric_label(metric: str, beta: float | None) -> str:
    if metric == "f1":
        return "F1 (Pit-stop class)"
    if metric == "fbeta":
        if beta is not None and abs(beta - round(beta)) < 1e-6:
            return f"F{int(round(beta))} (Pit-stop class)"
        if beta is not None:
            return f"Fbeta (beta={beta:g}) (Pit-stop class)"
        return "Fbeta (Pit-stop class)"
    if metric == "pr_auc":
        return "PR-AUC"
    if metric == "recall":
        return "Recall"
    if metric == "precision":
        return "Precision"
    return metric.upper()


def _metric_short(metric: str, beta: float | None) -> str:
    if metric == "f1":
        return "F1"
    if metric == "fbeta":
        if beta is not None and abs(beta - round(beta)) < 1e-6:
            return f"F{int(round(beta))}"
        if beta is not None:
            return f"Fbeta({beta:g})"
        return "Fbeta"
    if metric == "pr_auc":
        return "PR-AUC"
    if metric == "recall":
        return "Recall"
    if metric == "precision":
        return "Precision"
    return metric.upper()


def _add_header(fig, title: str, subtitle: str, strict_tag: bool) -> None:
    header_h = HEADER_H
    fig.patches.append(
        Rectangle(
            (0, 1 - header_h),
            1,
            header_h,
            transform=fig.transFigure,
            facecolor="#11141A",
            edgecolor="none",
            zorder=2,
        )
    )
    fig.text(
        0.02,
        1 - header_h / 2 + 0.01,
        _format_title(title),
        color=FG_COLOR,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        va="center",
        ha="left",
        zorder=3,
    )
    fig.text(
        0.02,
        1 - header_h + 0.01,
        subtitle,
        color=FG_COLOR,
        fontsize=SUBTITLE_SIZE,
        alpha=0.8,
        va="bottom",
        ha="left",
        zorder=3,
    )
    if strict_tag:
        tag_w = 0.08
        tag_h = 0.035
        x0 = 0.90
        y0 = 1 - header_h / 2 - tag_h / 2
        fig.patches.append(
            FancyBboxPatch(
                (x0, y0),
                tag_w,
                tag_h,
                transform=fig.transFigure,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=ACCENT,
                edgecolor="none",
                zorder=3,
            )
        )
        fig.text(
            x0 + tag_w / 2,
            y0 + tag_h / 2,
            "STRICT",
            color=FG_COLOR,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )


def _add_panel_card(ax) -> None:
    shadow = FancyBboxPatch(
        (0.01, -0.01),
        0.98,
        0.98,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#000000",
        edgecolor="none",
        alpha=0.35,
        zorder=0,
    )
    card = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=PANEL_COLOR,
        edgecolor="#1F2430",
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(shadow)
    ax.add_patch(card)
    ax.set_facecolor("none")


def _style_axis(ax, title: str) -> None:
    _add_panel_card(ax)
    ax.set_title(title.upper(), fontsize=AX_TITLE_SIZE, fontweight="bold", color=FG_COLOR, loc="left", pad=10)
    ax.grid(True, axis="y", color=FG_COLOR, alpha=GRID_ALPHA, linewidth=0.6)
    ax.tick_params(axis="both", colors=FG_COLOR, labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8A8A8")
    ax.spines["bottom"].set_color("#A8A8A8")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)


def _detect_stage_col(df: pd.DataFrame) -> str:
    stage_candidates = ["stage", "Stage", "stage_name", "stageName", "experiment", "stage_label"]
    col = _find_col(df, stage_candidates)
    if col is not None:
        return col
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if len(obj_cols) == 1:
        return obj_cols[0]
    cols_preview = ", ".join(df.columns.tolist())
    raise ValueError(
        "Cannot find stage label column. Expected one of: "
        f"{', '.join(stage_candidates)}.\nAvailable columns: {cols_preview}"
    )


def _extract_beta(df: pd.DataFrame) -> float | None:
    beta_col = _find_col(df, ["beta", "f_beta", "fbeta_beta"])
    if not beta_col:
        return None
    series = df[beta_col].dropna()
    if series.empty:
        return None
    try:
        return float(series.iloc[0])
    except (TypeError, ValueError):
        return None


def _metric_key(val: str) -> str | None:
    n = _normalize(str(val))
    if "f1" in n:
        return "f1"
    if "fbeta" in n or "f2" in n:
        return "fbeta"
    if "prauc" in n or "averageprecision" in n:
        return "pr_auc"
    if "precision" in n or n == "prec":
        return "precision"
    if "recall" in n or "tpr" in n or "sensitivity" in n:
        return "recall"
    return None


def _summary_from_long(df: pd.DataFrame, stage_col: str) -> pd.DataFrame | None:
    metric_col = _find_col(df, ["metric", "metric_name", "measure", "metric_type", "score_type"])
    mean_col = _find_col(df, ["mean", "metric_mean", "value", "score", "metric_value", "avg", "average"])
    std_col = _find_col(df, ["std", "metric_std", "stddev", "std_dev", "stdev", "sd"])
    if not metric_col or not mean_col:
        return None

    rows = []
    for (stage, metric_val), sub in df.groupby([stage_col, metric_col], dropna=False):
        key = _metric_key(metric_val)
        if key is None:
            continue
        vals = sub[mean_col].astype(float)
        mean_val = float(vals.mean())
        if std_col and std_col in sub.columns:
            std_val = float(sub[std_col].astype(float).mean())
        else:
            std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(
            {
                "stage": stage,
                f"mean_{key}": mean_val,
                f"std_{key}": std_val,
            }
        )
    if not rows:
        return None
    summary = pd.DataFrame(rows).groupby("stage", as_index=False).first()
    return summary


def _stage_tick_label(stage_num: int) -> str:
    if stage_num in (1, 2):
        group = "REFDATA"
    elif stage_num in (3, 4):
        group = "MYDATA+W"
    else:
        group = ""
    return f"S{stage_num}\n{group}" if group else f"S{stage_num}"


def _summary_from_wide(df: pd.DataFrame, stage_col: str) -> pd.DataFrame:
    metrics = ["f1", "fbeta", "precision", "recall", "pr_auc"]
    mean_aliases = {
        "f1": ["mean_f1", "f1_mean", "f1", "f1score", "f1_score"],
        "fbeta": ["mean_fbeta", "fbeta_mean", "fbeta", "f2", "f2_score", "f2mean"],
        "precision": ["mean_precision", "precision_mean", "precision", "prec"],
        "recall": ["mean_recall", "recall_mean", "recall", "tpr"],
        "pr_auc": ["mean_pr_auc", "pr_auc_mean", "pr_auc", "prauc", "average_precision"],
    }
    std_aliases = {
        "f1": ["std_f1", "f1_std", "std_f1score"],
        "fbeta": ["std_fbeta", "fbeta_std", "std_f2", "f2_std"],
        "precision": ["std_precision", "precision_std", "std_prec"],
        "recall": ["std_recall", "recall_std", "std_tpr"],
        "pr_auc": ["std_pr_auc", "pr_auc_std", "std_prauc"],
    }

    rows = []
    for _, r in df.iterrows():
        row = {"stage": r[stage_col]}
        for m in metrics:
            mean_col = _find_col(df, mean_aliases[m])
            if mean_col:
                row[f"mean_{m}"] = float(r[mean_col])
                std_col = _find_col(df, std_aliases[m])
                row[f"std_{m}"] = float(r[std_col]) if std_col else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def load_summary(df: pd.DataFrame) -> pd.DataFrame:
    stage_col = _detect_stage_col(df)
    summary = _summary_from_long(df, stage_col)
    if summary is None:
        summary = _summary_from_wide(df, stage_col)
    summary["stage_num"] = summary["stage"].apply(_stage_rank)
    summary["method"] = summary.apply(
        lambda r: _infer_method(str(r["stage"]), int(r["stage_num"]) if r["stage_num"] != 99 else None),
        axis=1,
    )
    summary = summary.sort_values(by="stage_num")
    return summary


def _metric_by_stage(summary: pd.DataFrame, metric: str) -> Dict[int, Tuple[float, float, str]]:
    mean_col = f"mean_{metric}"
    std_col = f"std_{metric}"
    if mean_col not in summary.columns:
        return {}
    out: Dict[int, Tuple[float, float, str]] = {}
    for _, r in summary.iterrows():
        stage_num = int(r["stage_num"])
        if stage_num == 99:
            continue
        mean_val = float(r[mean_col])
        std_val = float(r[std_col]) if std_col in summary.columns else 0.0
        out[stage_num] = (mean_val, std_val, str(r["method"]))
    return out


def _extract_fold_deltas(df: pd.DataFrame, stage_col: str, metric: str = "f1") -> Tuple[List[float], List[float]]:
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
        d21 = [float(s2[c]) - float(s1[c]) for _, c in fold_cols] if s1 is not None and s2 is not None else []
        d43 = [float(s4[c]) - float(s3[c]) for _, c in fold_cols] if s3 is not None and s4 is not None else []
        return d21, d43

    fold_col = _find_col(df, ["fold", "fold_id", "cv_fold"])
    metric_col = _find_col(df, [metric, f"{metric}_mean", f"mean_{metric}"])
    if fold_col and metric_col:
        pivot = df.pivot_table(index=fold_col, columns=stage_col, values=metric_col, aggfunc="mean")
        d21 = []
        d43 = []
        for fold in sorted(pivot.index.tolist()):
            row = pivot.loc[fold]
            s1 = s2 = s3 = s4 = None
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


def _save_fig(fig, outbase: Path) -> None:
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(outbase.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {outbase.with_suffix('.png')}")
    print(f"Saved: {outbase.with_suffix('.pdf')}")


def _plot_metric_panel(ax, metric_map: Dict[int, Tuple[float, float, str]], title: str) -> None:
    _style_axis(ax, title)
    if not metric_map:
        ax.text(0.5, 0.5, "Metric unavailable", color=FG_COLOR, fontsize=14, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    positions = {1: 0, 2: 1, 3: 2, 4: 3}
    xs = []
    ys = []
    yerr = []
    colors = []
    for stage_num in sorted(metric_map.keys()):
        mean_val, std_val, method = metric_map[stage_num]
        xs.append(positions[stage_num])
        ys.append(mean_val)
        yerr.append(std_val)
        colors.append(ACCENT if method == "MyMethod" else REF_COLOR)
    ax.bar(
        xs,
        ys,
        yerr=yerr,
        capsize=6,
        width=0.6,
        color=colors,
        edgecolor="#3A3A48",
        linewidth=1.0,
        error_kw={"ecolor": FG_COLOR, "elinewidth": 1.2, "alpha": 0.7},
        zorder=3,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([_stage_tick_label(n) for n in sorted(metric_map.keys())], fontsize=12, color=FG_COLOR)
    ax.tick_params(axis="x", pad=8)
    for x, v in zip(xs, ys):
        ax.text(x, min(1.0, v + 0.045), f"{v:.3f}", ha="center", va="bottom", color=FG_COLOR, fontsize=VALUE_SIZE)
    ax.axvline(1.5, color=FG_COLOR, alpha=0.12, linewidth=0.8)


def _plot_kpi_panel(
    ax,
    summary: pd.DataFrame,
    fold_d21: List[float],
    fold_d43: List[float],
    metric_key: str,
    metric_short: str,
) -> None:
    _style_axis(ax, "KPI DELTAS")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def _card(ax, xy, w, h, title, value, note=None):
        shadow = FancyBboxPatch(
            (xy[0] + 0.01, xy[1] - 0.01),
            w,
            h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#000000",
            edgecolor="none",
            alpha=0.35,
            zorder=2,
        )
        card = FancyBboxPatch(
            xy,
            w,
            h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#141822",
            edgecolor="#1F2430",
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(shadow)
        ax.add_patch(card)
        ax.text(xy[0] + 0.04, xy[1] + h - 0.12, title, transform=ax.transAxes, color=FG_COLOR, fontsize=12, alpha=0.8)
        ax.text(xy[0] + 0.04, xy[1] + h - 0.30, value, transform=ax.transAxes, color=ACCENT, fontsize=KPI_VALUE_SIZE, fontweight="bold")
        if note:
            ax.text(xy[0] + 0.04, xy[1] + 0.10, note, transform=ax.transAxes, color=FG_COLOR, fontsize=11, alpha=0.8)

    metric_map = _metric_by_stage(summary, metric_key)
    d21 = None
    d43 = None
    if 1 in metric_map and 2 in metric_map:
        d21 = metric_map[2][0] - metric_map[1][0]
    if 3 in metric_map and 4 in metric_map:
        d43 = metric_map[4][0] - metric_map[3][0]

    note21 = None
    if fold_d21:
        improved = sum(1 for d in fold_d21 if d > 0)
        note21 = f"Consistency: {improved}/{len(fold_d21)} folds improved"
    _card(
        ax,
        (0.06, 0.56),
        0.88,
        0.34,
        f"DELTA {metric_short} S2-S1",
        f"{d21:+.3f}" if d21 is not None else "n/a",
        note21,
    )

    note43 = None
    if fold_d43:
        improved = sum(1 for d in fold_d43 if d > 0)
        note43 = f"Consistency: {improved}/{len(fold_d43)} folds improved"
    _card(
        ax,
        (0.06, 0.12),
        0.88,
        0.34,
        f"DELTA {metric_short} S4-S3",
        f"{d43:+.3f}" if d43 is not None else "n/a",
        note43,
    )


def plot_figA(
    summary: pd.DataFrame,
    df_raw: pd.DataFrame,
    strict_tag: bool,
    outbase: Path,
    primary_metric: str,
    beta: float | None,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG_COLOR)
    _add_header(
        fig,
        "Supervisor Workflow - Leakage-Safe Results",
        "GroupKFold by race (unseen races), leakage-safe (_prev features)",
        strict_tag,
    )
    fig.text(
        0.02,
        0.02,
        "GroupKFold by race (unseen races), leakage-safe (_prev features)",
        color=FG_COLOR,
        fontsize=9,
        alpha=0.7,
        ha="left",
        va="bottom",
    )
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.95, bottom=0.06, top=0.84, wspace=0.12, hspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    metric_label = _format_metric_label(primary_metric, beta)
    metric_short = _metric_short(primary_metric, beta)
    _plot_metric_panel(ax1, _metric_by_stage(summary, primary_metric), metric_label)
    _plot_metric_panel(ax2, _metric_by_stage(summary, "pr_auc"), "PR-AUC")
    _plot_metric_panel(ax3, _metric_by_stage(summary, "recall"), "Recall")

    stage_col = _detect_stage_col(df_raw)
    fold_d21, fold_d43 = _extract_fold_deltas(df_raw, stage_col, metric=primary_metric)
    _plot_kpi_panel(ax4, summary, fold_d21, fold_d43, primary_metric, metric_short)

    _save_fig(fig, outbase)
    plt.close(fig)


def _gain_strip(ax, deltas: List[float], label: str, mean_fallback: float | None, note: str | None) -> None:
    _add_panel_card(ax)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8A8A8")
    ax.spines["bottom"].set_color("#A8A8A8")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=FG_COLOR, labelsize=12)

    if deltas:
        y = np.arange(1, len(deltas) + 1)
        ax.barh(y, deltas, color=ACCENT, edgecolor="#3A3A48", linewidth=0.8, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([f"Fold {i}" for i in y])
        mu = float(np.mean(deltas))
        all_pos = all(d > 0 for d in deltas)
    else:
        y = np.array([1])
        mu = float(mean_fallback) if mean_fallback is not None else 0.0
        ax.barh(y, [mu], color=ACCENT, edgecolor="#3A3A48", linewidth=0.8, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(["Mean"])
        all_pos = mu > 0

    ax.axvline(0.0, color=FG_COLOR, alpha=0.25, linewidth=1.0)
    ax.set_title(label.upper(), loc="left", color=FG_COLOR, fontsize=16, pad=8)

    card = FancyBboxPatch(
        (0.72, 0.70),
        0.26,
        0.20,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#141822",
        edgecolor="#1F2430",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(card)
    ax.text(0.75, 0.82, "Mean", transform=ax.transAxes, color=FG_COLOR, fontsize=11, alpha=0.8)
    ax.text(0.75, 0.74, f"{mu:+.3f}", transform=ax.transAxes, color=ACCENT, fontsize=18, fontweight="bold")

    if all_pos and deltas:
        ax.text(0.02, 0.90, "All folds improved", transform=ax.transAxes, color=FG_COLOR, fontsize=11, alpha=0.85)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, color=FG_COLOR, fontsize=10, alpha=0.7)

    if deltas:
        min_val = min(deltas)
        max_val = max(deltas)
    else:
        min_val = mu
        max_val = mu
    pad = max(0.02, max(abs(min_val), abs(max_val)) * 0.25)
    ax.set_xlim(min_val - pad, max_val + pad)


def plot_figB(
    summary: pd.DataFrame,
    df_raw: pd.DataFrame,
    strict_tag: bool,
    outbase: Path,
    primary_metric: str,
    beta: float | None,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)
    metric_short = _metric_short(primary_metric, beta)
    _add_header(
        fig,
        "Improvement Consistency",
        f"Delta {metric_short} across unseen races (GroupKFold by race)",
        strict_tag,
    )
    gs = fig.add_gridspec(2, 1, left=0.07, right=0.95, bottom=0.08, top=0.84, hspace=0.18)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    stage_col = _detect_stage_col(df_raw)
    fold_d21, fold_d43 = _extract_fold_deltas(df_raw, stage_col, metric=primary_metric)

    metric_map = _metric_by_stage(summary, primary_metric)
    mean_d21 = metric_map[2][0] - metric_map[1][0] if 1 in metric_map and 2 in metric_map else None
    mean_d43 = metric_map[4][0] - metric_map[3][0] if 3 in metric_map and 4 in metric_map else None

    note21 = None if fold_d21 else "mean-only (no fold detail available)"
    note43 = None if fold_d43 else "mean-only (no fold detail available)"

    _gain_strip(ax_top, fold_d21, f"Stage 2 minus Stage 1 ({metric_short} - RefData)", mean_d21, note21)
    _gain_strip(ax_bot, fold_d43, f"Stage 4 minus Stage 3 ({metric_short} - MyData+W)", mean_d43, note43)

    _save_fig(fig, outbase)
    plt.close(fig)


def plot_figC(
    summary: pd.DataFrame,
    strict_tag: bool,
    outbase: Path,
    primary_metric: str,
    beta: float | None,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    _add_header(
        fig,
        "Stage Comparison - Slide Friendly",
        f"GroupKFold by race (unseen races) - RefTech vs MyMethod ({_metric_short(primary_metric, beta)})",
        strict_tag,
    )
    ax = fig.add_axes([0.08, 0.14, 0.86, 0.68])
    _style_axis(ax, f"{_metric_short(primary_metric, beta)} DUMBBELL")

    metric_map = _metric_by_stage(summary, primary_metric)
    if not metric_map:
        ax.text(
            0.5,
            0.5,
            f"{_metric_short(primary_metric, beta)} unavailable",
            color=FG_COLOR,
            fontsize=14,
            ha="center",
            va="center",
        )
        _save_fig(fig, outbase)
        plt.close(fig)
        return

    def _val(stage_num: int) -> float | None:
        return metric_map[stage_num][0] if stage_num in metric_map else None

    pairs = [
        ("RefData", _val(1), _val(2)),
        ("MyData+W", _val(3), _val(4)),
    ]
    x_left, x_right = 0, 1
    for i, (label, v_ref, v_my) in enumerate(pairs):
        if v_ref is None or v_my is None:
            continue
        ax.plot([x_left, x_right], [v_ref, v_my], color=ACCENT, linewidth=3, alpha=0.9, zorder=3)
        ax.scatter([x_left], [v_ref], color=REF_COLOR, s=90, zorder=4)
        ax.scatter([x_right], [v_my], color=ACCENT, s=90, zorder=4)
        ax.text(x_left - 0.03, v_ref, f"{v_ref:.3f}", color=FG_COLOR, fontsize=14, ha="right", va="center")
        ax.text(x_right + 0.03, v_my, f"{v_my:.3f}", color=FG_COLOR, fontsize=14, ha="left", va="center")
        delta = v_my - v_ref
        ax.text(0.5, (v_ref + v_my) / 2 + 0.02, f"{delta:+.3f}", color=ACCENT, fontsize=14, fontweight="bold", ha="center")
        ax.text(0.5, (v_ref + v_my) / 2 - 0.04, label, color=FG_COLOR, fontsize=12, ha="center", alpha=0.8)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["REFTECH", "MYMETHOD"], color=FG_COLOR, fontsize=14, fontweight="bold")
    ax.set_ylabel(_format_metric_label(primary_metric, beta), color=FG_COLOR, fontsize=14)

    _save_fig(fig, outbase)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pit wall broadcast summary figures.")
    parser.add_argument("--in_csv", required=True, help="Path to stage_summary.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for pit wall figures")
    parser.add_argument(
        "--primary_metric",
        default="f1",
        help="Primary metric for comparisons: f1, fbeta, or f2.",
    )
    parser.add_argument("--beta", type=float, default=None, help="Beta value for Fbeta/F2 labeling.")
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        print(f"[ERROR] Input CSV not found: {in_csv}")
        sys.exit(1)

    try:
        df = pd.read_csv(in_csv)
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV: {in_csv}\n{exc}")
        sys.exit(1)

    if df.empty:
        print(f"[ERROR] CSV is empty: {in_csv}")
        sys.exit(1)

    fold_df = None
    fold_csv = in_csv.with_name("stage_folds_strict.csv")
    if fold_csv.exists():
        try:
            fold_df = pd.read_csv(fold_csv)
            if fold_df.empty:
                fold_df = None
        except Exception:
            fold_df = None

    try:
        summary = load_summary(df)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    primary_metric = args.primary_metric.strip().lower()
    if primary_metric == "f2":
        primary_metric = "fbeta"
    if primary_metric not in {"f1", "fbeta"}:
        print(f"[ERROR] Unsupported primary metric: {args.primary_metric}")
        sys.exit(1)

    beta = args.beta if args.beta is not None else _extract_beta(df)
    if primary_metric == "fbeta" and beta is None:
        beta = 2.0

    strict_csv = in_csv.with_name("stage_summary_strict.csv")
    strict_tag = "strict" in in_csv.stem.lower() or strict_csv.exists()
    df_raw = fold_df if fold_df is not None else df

    plot_figA(summary, df_raw, strict_tag, outdir / "pitwall_figA_overview", primary_metric, beta)
    plot_figB(summary, df_raw, strict_tag, outdir / "pitwall_figB_consistency", primary_metric, beta)
    plot_figC(summary, strict_tag, outdir / "pitwall_figC_slide", primary_metric, beta)


if __name__ == "__main__":
    main()
