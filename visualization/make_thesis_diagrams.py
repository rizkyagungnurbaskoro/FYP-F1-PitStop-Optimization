from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")


PALETTE = {
    "bg": "#0b0d12",
    "panel": "#121826",
    "edge": "#2a3342",
    "ink": "#f5f7fb",
    "muted": "#b2bccb",
    "accent": "#e10600",
    "accent2": "#ff9d2b",
    "accent3": "#17c3ff",
}


def _box(ax, x, y, w, h, text, fc, ec=None, fontsize=10, weight="bold"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec or PALETTE["edge"],
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=PALETTE["ink"],
        fontsize=fontsize,
        fontweight=weight,
    )


def _arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color=PALETTE["muted"],
    )
    ax.add_patch(arr)


def _base_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def make_system_architecture(outdir: Path, fmt: str, dpi: int) -> None:
    fig, ax = _base_axes((14, 3.5))
    ax.text(
        0.02,
        0.92,
        "System / Project Design",
        color=PALETTE["ink"],
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.86,
        "Decision-support pipeline for pit stop timing",
        color=PALETTE["muted"],
        fontsize=9,
    )
    stages = [
        ("Data", PALETTE["panel"]),
        ("Feature Eng", "#182033"),
        ("Model", "#1a243b"),
        ("Evaluation", "#1a2a40"),
        ("Dashboard", "#1b2f46"),
        ("Demo", "#1b354b"),
    ]
    x0 = 0.03
    gap = 0.015
    w = (0.94 - gap * (len(stages) - 1)) / len(stages)
    y = 0.38
    h = 0.30
    for i, (label, color) in enumerate(stages):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, label, color)
        if i < len(stages) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap * 0.9, y + h / 2)
    out_path = outdir / f"system_architecture.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_data_flow(outdir: Path, fmt: str, dpi: int) -> None:
    fig, ax = _base_axes((10, 4.2))
    ax.text(
        0.02,
        0.92,
        "Data Flow and Leakage Control",
        color=PALETTE["ink"],
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.86,
        "Strict split: GroupKFold by race with previous-lap features",
        color=PALETTE["muted"],
        fontsize=9,
    )
    _box(ax, 0.05, 0.60, 0.22, 0.18, "Race data", PALETTE["panel"])
    _box(ax, 0.34, 0.60, 0.22, 0.18, "Lap records", "#182033")
    _box(ax, 0.63, 0.60, 0.30, 0.18, "GroupKFold by race", "#1a2a40")
    _arrow(ax, 0.27, 0.69, 0.34, 0.69)
    _arrow(ax, 0.56, 0.69, 0.63, 0.69)
    _box(ax, 0.20, 0.22, 0.26, 0.18, "Train folds\n(unseen races)", "#1b2f46", fontsize=9)
    _box(ax, 0.54, 0.22, 0.26, 0.18, "Test fold\n(unseen races)", "#1b354b", fontsize=9)
    _arrow(ax, 0.72, 0.60, 0.34, 0.40)
    _arrow(ax, 0.78, 0.60, 0.64, 0.40)
    _box(ax, 0.05, 0.36, 0.18, 0.14, "Use *_prev\nfeatures only", "#1f2735", fontsize=8)
    _box(ax, 0.77, 0.36, 0.18, 0.14, "Leakage guard\nsame-lap removed", "#1f2735", fontsize=8)
    out_path = outdir / f"data_flow_leakage.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_model_pipeline(outdir: Path, fmt: str, dpi: int) -> None:
    fig, ax = _base_axes((12, 4.2))
    ax.text(
        0.02,
        0.92,
        "Model Pipeline (Method)",
        color=PALETTE["ink"],
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.86,
        "XGBoost with calibration and threshold tuning",
        color=PALETTE["muted"],
        fontsize=9,
    )
    blocks = [
        ("Feature matrix", PALETTE["panel"]),
        ("One-hot encode", "#182033"),
        ("XGBoost", "#1a243b"),
        ("Calibrate", "#1a2a40"),
        ("Tune threshold", "#1b2f46"),
        ("Decision call", "#1b354b"),
    ]
    x0 = 0.03
    gap = 0.018
    w = (0.80 - gap * (len(blocks) - 1)) / len(blocks)
    y = 0.42
    h = 0.24
    for i, (label, color) in enumerate(blocks):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, label, color, fontsize=9)
        if i < len(blocks) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap * 0.9, y + h / 2)
    _box(ax, 0.84, 0.28, 0.13, 0.46, "Stages\nS1-S4", "#1f2735", fontsize=10)
    ax.text(
        0.84,
        0.24,
        "Fair comparison",
        color=PALETTE["muted"],
        fontsize=8,
    )
    out_path = outdir / f"model_pipeline.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis diagrams.")
    parser.add_argument(
        "--outdir",
        default=str(Path("results") / "summary_plots" / "diagrams"),
        help="Output directory for diagram images.",
    )
    parser.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    make_system_architecture(outdir, args.fmt, args.dpi)
    make_data_flow(outdir, args.fmt, args.dpi)
    make_model_pipeline(outdir, args.fmt, args.dpi)

    print(f"[OK] Saved diagrams to {outdir}")


if __name__ == "__main__":
    main()
