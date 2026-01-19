from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.exp_config import get_paths
from experiments.exp_utils import add_feature_engineering, build_feature_list, ensure_dir, load_csv, safe_numeric

TARGET_COL = "decide_pitstop"
GROUP_COL = "race_id"
TOP_CATS = 10
HIST_BINS = 10


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return 0.0

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n = a_sorted.size
    m = b_sorted.size
    i = 0
    j = 0
    d = 0.0
    while i < n and j < m:
        if a_sorted[i] <= b_sorted[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / n - j / m))
    if i < n:
        d = max(d, abs(1.0 - j / m))
    if j < m:
        d = max(d, abs(i / n - 1.0))
    return float(d)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = np.clip(m, eps, 1.0)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def _class_balance(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y = df[TARGET_COL].astype(int)
    overall = {
        "dataset": label,
        "n": int(len(y)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "pos_rate": float((y == 1).mean()) if len(y) else 0.0,
    }

    by_group = (
        df.groupby(GROUP_COL, dropna=False)[TARGET_COL]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={GROUP_COL: "group_id", "count": "n", "sum": "pos"})
    )
    by_group["neg"] = by_group["n"] - by_group["pos"]
    by_group["pos_rate"] = np.where(by_group["n"] > 0, by_group["pos"] / by_group["n"], 0.0)
    by_group["dataset"] = label

    stats = {
        "dataset": label,
        "pos_rate_min": float(by_group["pos_rate"].min()) if not by_group.empty else 0.0,
        "pos_rate_median": float(by_group["pos_rate"].median()) if not by_group.empty else 0.0,
        "pos_rate_max": float(by_group["pos_rate"].max()) if not by_group.empty else 0.0,
        "groups": int(by_group.shape[0]),
    }

    return pd.DataFrame([overall]), by_group, stats


def _missingness(df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in features:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "dataset": label,
                "feature": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_pct": float(missing / n) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "feature"], ascending=[False, True])


def _numeric_drift(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    features: list[str],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    rows = []
    for col in features:
        if df_a[col].dtype == "object" or df_b[col].dtype == "object":
            continue
        a = pd.to_numeric(df_a[col], errors="coerce").values
        b = pd.to_numeric(df_b[col], errors="coerce").values
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0:
            continue
        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        std_a = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
        std_b = float(np.std(b, ddof=1)) if b.size > 1 else 0.0
        pooled = np.sqrt((std_a**2 + std_b**2) / 2.0) if (std_a or std_b) else 0.0
        smd = float((mean_a - mean_b) / pooled) if pooled > 0 else 0.0
        ks = _ks_statistic(a, b)

        hist_min = float(np.nanmin([a.min(), b.min()]))
        hist_max = float(np.nanmax([a.max(), b.max()]))
        if hist_min == hist_max:
            js = 0.0
        else:
            bins = np.linspace(hist_min, hist_max, HIST_BINS + 1)
            ha, _ = np.histogram(a, bins=bins)
            hb, _ = np.histogram(b, bins=bins)
            js = _js_divergence(ha.astype(float), hb.astype(float))

        rows.append(
            {
                "feature": col,
                f"mean_{label_a}": mean_a,
                f"mean_{label_b}": mean_b,
                f"std_{label_a}": std_a,
                f"std_{label_b}": std_b,
                "smd": smd,
                "ks": ks,
                "js_div": js,
            }
        )
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(["smd", "ks"], ascending=[False, False])
    return df_out


def _categorical_drift(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    features: list[str],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    rows = []
    for col in features:
        if df_a[col].dtype != "object" and df_b[col].dtype != "object":
            continue
        a = df_a[col].astype("object").fillna("NA")
        b = df_b[col].astype("object").fillna("NA")
        top_a = a.value_counts(dropna=False).head(TOP_CATS)
        top_b = b.value_counts(dropna=False).head(TOP_CATS)
        cats = set(top_a.index).union(set(top_b.index))
        total_a = float(len(a))
        total_b = float(len(b))
        if total_a == 0 or total_b == 0:
            continue
        tvd = 0.0
        for cat in cats:
            pa = float(top_a.get(cat, 0.0) / total_a)
            pb = float(top_b.get(cat, 0.0) / total_b)
            tvd += abs(pa - pb)
        tvd *= 0.5
        rows.append(
            {
                "feature": col,
                f"unique_{label_a}": int(a.nunique(dropna=False)),
                f"unique_{label_b}": int(b.nunique(dropna=False)),
                "top_cat_tvd": float(tvd),
            }
        )
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(["top_cat_tvd", "feature"], ascending=[False, True])
    return df_out


def main() -> None:
    paths = get_paths()
    out_dir = Path(paths.out_summary_plots)
    ensure_dir(out_dir)

    df_ref = add_feature_engineering(safe_numeric(load_csv(paths.ref_csv)))
    df_my = add_feature_engineering(safe_numeric(load_csv(paths.my_csv)))
    df_my_w = add_feature_engineering(safe_numeric(load_csv(paths.my_weather_csv)))

    _require_columns(df_ref, [TARGET_COL, GROUP_COL], "Reference dataset")
    _require_columns(df_my, [TARGET_COL, GROUP_COL], "Personal dataset")
    _require_columns(df_my_w, [TARGET_COL, GROUP_COL], "Stage 3/4 dataset")

    feats_ref = build_feature_list(df_ref, TARGET_COL, GROUP_COL)
    feats_my = build_feature_list(df_my, TARGET_COL, GROUP_COL)
    feats_my_w = build_feature_list(df_my_w, TARGET_COL, GROUP_COL)
    common_features = sorted(set(feats_ref).intersection(set(feats_my_w)))

    overall_rows = []
    by_group_rows = []
    stats_rows = []
    for df, label in [(df_ref, "ref"), (df_my, "my"), (df_my_w, "my_w")]:
        overall, by_group, stats = _class_balance(df, label)
        overall_rows.append(overall)
        by_group_rows.append(by_group)
        stats_rows.append(stats)

        missing = _missingness(df, build_feature_list(df, TARGET_COL, GROUP_COL), label)
        missing.to_csv(out_dir / f"diagnostic_missing_{label}.csv", index=False)

    pd.concat(overall_rows, ignore_index=True).to_csv(
        out_dir / "diagnostic_class_balance_overall.csv", index=False
    )
    pd.concat(by_group_rows, ignore_index=True).to_csv(
        out_dir / "diagnostic_class_balance_by_group.csv", index=False
    )
    pd.DataFrame(stats_rows).to_csv(
        out_dir / "diagnostic_class_balance_stats.csv", index=False
    )

    drift_num = _numeric_drift(df_ref, df_my_w, common_features, "ref", "my_w")
    drift_num.to_csv(out_dir / "diagnostic_drift_numeric_ref_vs_my_w.csv", index=False)

    drift_cat = _categorical_drift(df_ref, df_my_w, common_features, "ref", "my_w")
    drift_cat.to_csv(out_dir / "diagnostic_drift_categorical_ref_vs_my_w.csv", index=False)

    summary = {
        "features_ref": int(len(feats_ref)),
        "features_my": int(len(feats_my)),
        "features_my_w": int(len(feats_my_w)),
        "features_common_ref_my_w": int(len(common_features)),
    }
    (out_dir / "diagnostic_summary.json").write_text(
        pd.Series(summary).to_json(indent=2),
        encoding="utf-8",
    )

    print("[OK] Diagnostics saved in results/summary_plots/")


if __name__ == "__main__":
    main()
