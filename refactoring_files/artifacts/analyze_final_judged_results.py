#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_ROOT = Path("refactoring_files") / "artifacts" / "Final_judged_files"
DEFAULT_OUT = DEFAULT_ROOT / "combined_analysis"

BOOK_ORDER = ["NEWS", "SCIFI", "THRILLER"]
METHOD_ORDER = ["DP", "RAG"]
CONDITION_ORDER = [f"{b}_{m}" for b in BOOK_ORDER for m in METHOD_ORDER]
CATEGORY_ORDER = [
    "Search",
    "Stateful Processing",
    "Recall and Edit",
    "Composite",
    "Match and Compare",
    "Spot the Differences",
    "Compute on Sets and Lists",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine judged episodic-memory outputs and generate analysis tables + plots."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mean_ci(values: Iterable[float], n_boot: int, seed: int) -> tuple[float, float, float, np.ndarray]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        nan = float("nan")
        return nan, nan, nan, np.array([])
    rng = np.random.default_rng(seed)
    if arr.size == 1:
        m = float(arr[0])
        return m, m, m, np.repeat(m, n_boot)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi), boot


def metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("f1", "size"),
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            exact_match_rate=("exact_match", "mean"),
        )
        .reset_index()
    )
    return out


def discover_rows(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/*/judged_merged_rows.csv"))
    if not files:
        raise FileNotFoundError(f"No judged_merged_rows.csv found under: {root}")

    chunks: list[pd.DataFrame] = []
    for file in files:
        condition = file.parent.name
        model = file.parent.parent.name
        if "_" not in condition:
            continue
        book, method = condition.split("_", 1)
        df = pd.read_csv(file)
        df["model"] = model
        df["condition"] = condition
        df["book"] = book
        df["method"] = method
        if "row_key" not in df.columns:
            if "row_idx" in df.columns:
                df["row_key"] = df["row_idx"].astype(str).map(lambda x: f"row_idx:{x}")
            else:
                df["row_key"] = np.arange(len(df)).astype(str).map(lambda x: f"row_idx:{x}")
        for col in ["f1", "precision", "recall", "exact_match", "row_idx"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["row_key"] = df["row_key"].astype(str)
        df["minerva_category"] = (
            df["minerva_category"].astype(str).replace({"nan": "unknown"})
            if "minerva_category" in df.columns
            else "unknown"
        )
        chunks.append(df)

    rows = pd.concat(chunks, ignore_index=True)
    return rows


def paired_differences(rows: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (model, book), g in rows.groupby(["model", "book"], dropna=False):
        dp = g[g["method"] == "DP"].copy()
        rag = g[g["method"] == "RAG"].copy()
        if dp.empty or rag.empty:
            continue
        dp = dp[["row_key", "f1", "exact_match"]].drop_duplicates("row_key")
        rag = rag[["row_key", "f1", "exact_match"]].drop_duplicates("row_key")
        merged = pd.merge(dp, rag, on="row_key", how="inner", suffixes=("_dp", "_rag"))
        if merged.empty:
            continue
        merged["diff_f1"] = merged["f1_dp"] - merged["f1_rag"]
        merged["diff_em"] = merged["exact_match_dp"] - merged["exact_match_rag"]
        merged["model"] = model
        merged["book"] = book
        merged["n_dp"] = len(dp)
        merged["n_rag"] = len(rag)
        merged["n_paired"] = len(merged)
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def heatmap(
    pivot: pd.DataFrame,
    title: str,
    out_path: Path,
    cmap: str = "YlGnBu",
    center: float | None = None,
) -> None:
    if pivot.empty:
        return
    plt.figure(figsize=(max(8, 1.2 * pivot.shape[1]), max(4, 0.55 * pivot.shape[0] + 2)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar_kws={"shrink": 0.9},
        linewidths=0.5,
        linecolor="white",
        center=center,
    )
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    title: str,
    out_path: Path,
    order: list[str] | None = None,
    hue_order: list[str] | None = None,
    rotate_xticks: int = 0,
) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=df, x=x, y=y, hue=hue, order=order, hue_order=hue_order, errorbar=None)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean F1")
    ax.set_xlabel("")
    if rotate_xticks:
        plt.xticks(rotation=rotate_xticks, ha="right")
    if hue is not None:
        plt.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def lineplot_category_by_method(
    by_category: pd.DataFrame,
    method: str,
    model_order: list[str],
    out_path: Path,
) -> None:
    sub = by_category[by_category["method"] == method].copy()
    if sub.empty:
        return
    categories = [c for c in CATEGORY_ORDER if c in set(sub["minerva_category"])]
    if "unknown" in set(sub["minerva_category"]):
        categories.append("unknown")
    sub["minerva_category"] = pd.Categorical(sub["minerva_category"], categories=categories, ordered=True)
    sub = sub.sort_values(["minerva_category", "model"])

    plt.figure(figsize=(11, 5.8))
    ax = sns.lineplot(
        data=sub,
        x="minerva_category",
        y="mean_f1",
        hue="model",
        hue_order=model_order,
        marker="o",
        linewidth=2,
    )
    ax.set_title(f"Category-wise F1 by Model ({method})")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Mean F1")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _chrono_components(df: pd.DataFrame) -> tuple[float, float, float]:
    retrieval = df.get("retrieval_type", pd.Series("", index=df.index)).astype(str).str.lower()
    get_mode = df.get("get", pd.Series("", index=df.index)).astype(str).str.lower()
    latest_mask = retrieval.str.contains("latest", na=False) | get_mode.eq("latest")
    chrono_mask = retrieval.str.contains("chron", na=False) | get_mode.str.contains("chron", na=False)
    latest = float(df.loc[latest_mask, "f1"].mean()) if latest_mask.any() else float("nan")
    chrono = float(df.loc[chrono_mask, "f1"].mean()) if chrono_mask.any() else float("nan")
    vals = [x for x in [latest, chrono] if not math.isnan(x)]
    awareness = float(sum(vals) / len(vals)) if vals else float("nan")
    return latest, chrono, awareness


def chronological_scores(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_method_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for (model, method), g in rows.groupby(["model", "method"], dropna=False):
        latest, chrono, awareness = _chrono_components(g)
        retrieval = g.get("retrieval_type", pd.Series("", index=g.index)).astype(str).str.lower()
        get_mode = g.get("get", pd.Series("", index=g.index)).astype(str).str.lower()
        n_latest = int((retrieval.str.contains("latest", na=False) | get_mode.eq("latest")).sum())
        n_chrono = int((retrieval.str.contains("chron", na=False) | get_mode.str.contains("chron", na=False)).sum())
        model_method_rows.append(
            {
                "model": model,
                "method": method,
                "n_rows": len(g),
                "n_latest_rows": n_latest,
                "n_chronological_rows": n_chrono,
                "latest_state_score": latest,
                "chronological_order_score": chrono,
                "chronological_awareness_score": awareness,
            }
        )

    for model, g in rows.groupby("model", dropna=False):
        latest, chrono, awareness = _chrono_components(g)
        retrieval = g.get("retrieval_type", pd.Series("", index=g.index)).astype(str).str.lower()
        get_mode = g.get("get", pd.Series("", index=g.index)).astype(str).str.lower()
        n_latest = int((retrieval.str.contains("latest", na=False) | get_mode.eq("latest")).sum())
        n_chrono = int((retrieval.str.contains("chron", na=False) | get_mode.str.contains("chron", na=False)).sum())
        model_rows.append(
            {
                "model": model,
                "n_rows": len(g),
                "n_latest_rows": n_latest,
                "n_chronological_rows": n_chrono,
                "latest_state_score": latest,
                "chronological_order_score": chrono,
                "chronological_awareness_score": awareness,
            }
        )

    return pd.DataFrame(model_method_rows), pd.DataFrame(model_rows)


def forest_with_ci(
    table: pd.DataFrame,
    label_col: str,
    mean_col: str,
    lo_col: str,
    hi_col: str,
    title: str,
    out_path: Path,
    x_label: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    if table.empty:
        return
    plot = table.copy().sort_values(mean_col, ascending=True)
    y = np.arange(len(plot))
    means = plot[mean_col].to_numpy(dtype=float)
    los = plot[lo_col].to_numpy(dtype=float)
    his = plot[hi_col].to_numpy(dtype=float)
    err_low = means - los
    err_high = his - means

    plt.figure(figsize=(9, max(3.2, 0.6 * len(plot) + 1.4)))
    plt.errorbar(
        means,
        y,
        xerr=[err_low, err_high],
        fmt="o",
        color="#1f77b4",
        ecolor="black",
        elinewidth=1.6,
        capsize=3,
    )
    plt.yticks(y, plot[label_col].tolist())
    if xlim is None:
        x_min = float(np.nanmin(np.concatenate([means, los, his])))
        x_max = float(np.nanmax(np.concatenate([means, los, his])))
        span = x_max - x_min
        pad = 0.08 * span if span > 1e-9 else 0.05
        plt.xlim(x_min - pad, x_max + pad)
    else:
        plt.xlim(xlim[0], xlim[1])
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(x_label)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def pair_outcomes(
    pairs: pd.DataFrame,
    tol: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    p = pairs.copy()
    delta = p["f1_rag"] - p["f1_dp"]
    p["delta_f1_rag_minus_dp"] = delta
    p["pair_outcome"] = np.where(delta > tol, "RAG wins", np.where(delta < -tol, "RAG loses", "Tie"))

    def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        counts = (
            df.groupby(group_cols + ["pair_outcome"], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        if group_cols:
            counts["pct"] = counts.groupby(group_cols, dropna=False)["n"].transform(lambda s: 100.0 * s / s.sum())
        else:
            total = float(counts["n"].sum()) if len(counts) else 1.0
            counts["pct"] = 100.0 * counts["n"] / total
        return counts.sort_values(group_cols + ["pair_outcome"])

    overall = summarize(p, [])
    by_model = summarize(p, ["model"])
    by_model_book = summarize(p, ["model", "book"])
    return overall, by_model, by_model_book


def stacked_percent_bar_from_counts(
    counts: pd.DataFrame,
    index_cols: list[str],
    category_col: str,
    order: list[str] | None,
    category_order: list[str],
    title: str,
    out_path: Path,
) -> None:
    if counts.empty:
        return
    c = counts.copy()
    idx_name = "__idx__"
    if len(index_cols) == 1:
        c[idx_name] = c[index_cols[0]].astype(str)
    else:
        c[idx_name] = c[index_cols].astype(str).agg(" | ".join, axis=1)

    pivot = c.pivot_table(index=idx_name, columns=category_col, values="pct", aggfunc="sum", fill_value=0)
    existing_cat_order = [x for x in category_order if x in pivot.columns]
    pivot = pivot.reindex(columns=existing_cat_order)
    if order is not None:
        pivot = pivot.reindex([o for o in order if o in pivot.index])

    colors = ["#2ca02c", "#7f7f7f", "#d62728"]  # win, tie, lose
    plt.figure(figsize=(max(8, 0.9 * len(pivot.index) + 2), 5.4))
    bottom = np.zeros(len(pivot.index))
    x = np.arange(len(pivot.index))
    for i, cat in enumerate(existing_cat_order):
        vals = pivot[cat].to_numpy(dtype=float)
        plt.bar(x, vals, bottom=bottom, label=cat, color=colors[i % len(colors)], width=0.75)
        bottom += vals

    plt.xticks(x, pivot.index.tolist(), rotation=20 if len(pivot.index) > 5 else 0, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Percentage of Paired Questions")
    plt.title(title)
    plt.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def ecdf_delta_plot(pairs: pd.DataFrame, model_order: list[str], out_path: Path) -> None:
    if pairs.empty:
        return
    plt.figure(figsize=(10, 6))
    for model in model_order:
        sub = pairs[pairs["model"] == model]
        if sub.empty:
            continue
        x = np.sort((sub["f1_rag"] - sub["f1_dp"]).to_numpy(dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", linewidth=2, label=f"{model} (n={len(x)})")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlim(-1.01, 1.01)
    plt.ylim(0, 1.01)
    plt.xlabel("Per-question Delta F1 (RAG - DP)")
    plt.ylabel("ECDF")
    plt.title("ECDF of Delta F1 (RAG - DP) by Model")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def retrieval_delta_table(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    df["retrieval_type"] = df.get("retrieval_type", pd.Series("", index=df.index)).astype(str)
    grouped = (
        df.groupby(["model", "method", "retrieval_type"], dropna=False)
        .agg(n_rows=("f1", "size"), sum_f1=("f1", "sum"))
        .reset_index()
    )
    grouped["weighted_mean_f1"] = grouped["sum_f1"] / grouped["n_rows"].clip(lower=1)

    rag = grouped[grouped["method"] == "RAG"][["model", "retrieval_type", "n_rows", "weighted_mean_f1"]].rename(
        columns={"n_rows": "n_rag", "weighted_mean_f1": "f1_rag_weighted"}
    )
    dp = grouped[grouped["method"] == "DP"][["model", "retrieval_type", "n_rows", "weighted_mean_f1"]].rename(
        columns={"n_rows": "n_dp", "weighted_mean_f1": "f1_dp_weighted"}
    )
    merged = pd.merge(rag, dp, on=["model", "retrieval_type"], how="outer")
    merged["f1_rag_weighted"] = pd.to_numeric(merged["f1_rag_weighted"], errors="coerce")
    merged["f1_dp_weighted"] = pd.to_numeric(merged["f1_dp_weighted"], errors="coerce")
    merged["delta_rag_minus_dp"] = merged["f1_rag_weighted"] - merged["f1_dp_weighted"]
    return merged.sort_values(["model", "retrieval_type"])


def error_bucket(row: pd.Series) -> str:
    try:
        f1 = float(row.get("f1", np.nan))
    except Exception:
        f1 = np.nan
    em = int(row.get("exact_match", 0)) if not pd.isna(row.get("exact_match", np.nan)) else 0
    if em == 1:
        return "Exact Match"
    if not math.isnan(f1) and abs(f1) < 1e-12:
        return "Zero F1"
    if not math.isnan(f1) and f1 < 0.5:
        return "Low F1 (0,0.5)"
    return "Mid F1 [0.5,1)"


def stacked_error_profile(
    rows: pd.DataFrame,
    out_dir: Path,
    model_order: list[str],
) -> None:
    bucket_order = ["Exact Match", "Mid F1 [0.5,1)", "Low F1 (0,0.5)", "Zero F1"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    df = rows.copy()
    df["error_bucket"] = df.apply(error_bucket, axis=1)

    # Mode-only stacked bars
    mode_counts = (
        df.groupby(["method", "error_bucket"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    mode_counts["pct"] = mode_counts.groupby("method", dropna=False)["n"].transform(lambda s: 100.0 * s / s.sum())
    mode_counts.to_csv(out_dir / "error_profile_by_method.csv", index=False)

    pivot = mode_counts.pivot_table(index="method", columns="error_bucket", values="pct", fill_value=0)
    pivot = pivot.reindex(index=METHOD_ORDER, columns=[b for b in bucket_order if b in pivot.columns])
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    plt.figure(figsize=(7.8, 5.2))
    for i, b in enumerate(pivot.columns):
        vals = pivot[b].to_numpy(dtype=float)
        plt.bar(x, vals, bottom=bottom, width=0.65, label=b, color=colors[i % len(colors)])
        bottom += vals
    plt.xticks(x, pivot.index.tolist())
    plt.ylim(0, 100)
    plt.ylabel("Percentage of Questions")
    plt.title("Error Profile by Mode")
    plt.legend(title="Bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "stacked_error_profile_by_method.png", dpi=180)
    plt.close()

    # Model x mode stacked bars
    df["model_method"] = df["model"].astype(str) + " | " + df["method"].astype(str)
    mm_counts = (
        df.groupby(["model", "method", "model_method", "error_bucket"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    mm_counts["pct"] = mm_counts.groupby(["model", "method"], dropna=False)["n"].transform(
        lambda s: 100.0 * s / s.sum()
    )
    mm_counts.to_csv(out_dir / "error_profile_by_model_method.csv", index=False)

    model_method_order = []
    for m in model_order:
        for method in METHOD_ORDER:
            label = f"{m} | {method}"
            if label in set(mm_counts["model_method"]):
                model_method_order.append(label)

    pivot = mm_counts.pivot_table(index="model_method", columns="error_bucket", values="pct", fill_value=0)
    pivot = pivot.reindex(index=model_method_order, columns=[b for b in bucket_order if b in pivot.columns])
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    plt.figure(figsize=(max(10, 0.95 * len(pivot.index) + 2), 5.6))
    for i, b in enumerate(pivot.columns):
        vals = pivot[b].to_numpy(dtype=float)
        plt.bar(x, vals, bottom=bottom, width=0.75, label=b, color=colors[i % len(colors)])
        bottom += vals
    plt.xticks(x, pivot.index.tolist(), rotation=25, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Percentage of Questions")
    plt.title("Error Profile by Model and Mode")
    plt.legend(title="Bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "stacked_error_profile_by_model_method.png", dpi=180)
    plt.close()


def gardner_altman_by_group(
    pairs: pd.DataFrame,
    group_col: str,
    out_path: Path,
    title: str,
    n_boot: int,
    seed: int,
) -> None:
    if pairs.empty:
        return
    groups = list(pairs[group_col].dropna().unique())
    if not groups:
        return

    fig = plt.figure(figsize=(12, 3.4 * len(groups)), constrained_layout=True)
    gs = fig.add_gridspec(len(groups), 2, width_ratios=[1.05, 1.2], wspace=0.22, hspace=0.38)

    for i, g in enumerate(groups):
        sub = pairs[pairs[group_col] == g].copy()
        if sub.empty:
            continue
        ax_l = fig.add_subplot(gs[i, 0])
        ax_r = fig.add_subplot(gs[i, 1])

        rng = np.random.default_rng(seed + i)
        x_rag = rng.normal(0, 0.06, len(sub))
        x_dp = rng.normal(1, 0.06, len(sub))
        ax_l.scatter(x_rag, sub["f1_rag"], s=12, alpha=0.35, label="RAG", color="#1f77b4")
        ax_l.scatter(x_dp, sub["f1_dp"], s=12, alpha=0.35, label="DP", color="#ff7f0e")
        m_rag = sub["f1_rag"].mean()
        m_dp = sub["f1_dp"].mean()
        ax_l.plot([0, 1], [m_rag, m_dp], color="black", linewidth=2, marker="o")
        ax_l.set_xticks([0, 1], ["RAG", "DP"])
        ax_l.set_ylim(-0.02, 1.02)
        ax_l.set_ylabel("F1")
        ax_l.set_title(f"{g} (n={len(sub)})")
        if i == 0:
            ax_l.legend(loc="lower right")

        diff_mean, diff_lo, diff_hi, boot = mean_ci(sub["diff_f1"], n_boot=n_boot, seed=seed + 1000 + i)
        if boot.size > 0:
            sns.kdeplot(x=boot, ax=ax_r, color="#2ca02c", fill=True, alpha=0.35, linewidth=1.0)
        ax_r.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        if not math.isnan(diff_mean):
            ax_r.axvline(diff_mean, color="black", linewidth=2)
            ax_r.axvline(diff_lo, color="black", linestyle=":", linewidth=1.3)
            ax_r.axvline(diff_hi, color="black", linestyle=":", linewidth=1.3)
            ax_r.set_title(f"Delta(DP-RAG)={diff_mean:.3f} [{diff_lo:.3f}, {diff_hi:.3f}]")
        ax_r.set_xlabel("Mean Difference in F1 (DP - RAG)")
        ax_r.set_ylabel("Density")

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    overall: pd.DataFrame,
    by_model: pd.DataFrame,
    by_method: pd.DataFrame,
    by_book_method: pd.DataFrame,
    by_category: pd.DataFrame,
    diffs_model: pd.DataFrame,
    diffs_book: pd.DataFrame,
    missing_notes: list[str],
) -> None:
    top_model = by_model.sort_values("mean_f1", ascending=False).iloc[0]
    bottom_model = by_model.sort_values("mean_f1", ascending=True).iloc[0]
    rag_mean = by_method.loc[by_method["method"] == "RAG", "mean_f1"].iloc[0]
    dp_mean = by_method.loc[by_method["method"] == "DP", "mean_f1"].iloc[0]
    best_book_method = by_book_method.sort_values("mean_f1", ascending=False).iloc[0]
    hardest_book_method = by_book_method.sort_values("mean_f1", ascending=True).iloc[0]

    cat_mean = by_category.groupby("minerva_category", as_index=False)["mean_f1"].mean()
    easiest_cat = cat_mean.sort_values("mean_f1", ascending=False).iloc[0]
    hardest_cat = cat_mean.sort_values("mean_f1", ascending=True).iloc[0]

    lines = [
        "# Final Episodic Memory Benchmark Analysis",
        "",
        "## Coverage",
        f"- Total evaluated rows: {int(overall['n_rows'].sum())}",
        f"- Model-book-method conditions: {len(overall)}",
        "",
        "## Leaderboard",
        f"- Best model by mean F1: **{top_model['model']}** ({top_model['mean_f1']:.4f}, EM {top_model['exact_match_rate']:.4f})",
        f"- Lowest model by mean F1: **{bottom_model['model']}** ({bottom_model['mean_f1']:.4f}, EM {bottom_model['exact_match_rate']:.4f})",
        "",
        "## Method Effect",
        f"- Global mean F1 (RAG): {rag_mean:.4f}",
        f"- Global mean F1 (DP): {dp_mean:.4f}",
        f"- Net delta (RAG - DP): {rag_mean - dp_mean:+.4f}",
        "",
        "## Best/Worst Conditions",
        f"- Best condition: **{best_book_method['book']}_{best_book_method['method']}** (mean F1 {best_book_method['mean_f1']:.4f})",
        f"- Hardest condition: **{hardest_book_method['book']}_{hardest_book_method['method']}** (mean F1 {hardest_book_method['mean_f1']:.4f})",
        "",
        "## Category Behavior",
        f"- Easiest category (avg over all models/methods): **{easiest_cat['minerva_category']}** ({easiest_cat['mean_f1']:.4f})",
        f"- Hardest category (avg over all models/methods): **{hardest_cat['minerva_category']}** ({hardest_cat['mean_f1']:.4f})",
        "",
        "## DP vs RAG (Estimation)",
        "- See Gardner-Altman estimation plots for model-level and book-level effects.",
        "",
    ]
    if not diffs_model.empty:
        lines.append("### Mean Delta F1 (DP - RAG) By Model")
        for _, r in diffs_model.sort_values("mean_diff_f1", ascending=False).iterrows():
            lines.append(
                f"- {r['model']}: {r['mean_diff_f1']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] (paired n={int(r['n'])})"
            )
        lines.append("")

    if not diffs_book.empty:
        lines.append("### Mean Delta F1 (DP - RAG) By Book")
        for _, r in diffs_book.sort_values("mean_diff_f1", ascending=False).iterrows():
            lines.append(
                f"- {r['book']}: {r['mean_diff_f1']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] (paired n={int(r['n'])})"
            )
        lines.append("")

    if missing_notes:
        lines.append("## Data Notes")
        for note in missing_notes:
            lines.append(f"- {note}")
        lines.append("")

    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")

    rows = discover_rows(root)
    rows.to_csv(out_dir / "combined_judged_rows.csv", index=False)

    # Main summary tables
    overall = metric_summary(rows, ["model", "book", "method", "condition"]).sort_values(
        ["model", "book", "method"]
    )
    by_model = metric_summary(rows, ["model"]).sort_values("mean_f1", ascending=False)
    by_method = metric_summary(rows, ["method"]).sort_values("mean_f1", ascending=False)
    by_book = metric_summary(rows, ["book"]).sort_values("mean_f1", ascending=False)
    by_book_method = metric_summary(rows, ["book", "method"]).sort_values(
        ["book", "method"]
    )
    by_model_method = metric_summary(rows, ["model", "method"]).sort_values(
        ["model", "method"]
    )
    by_retrieval = metric_summary(rows, ["model", "method", "retrieval_type"]).sort_values(
        ["model", "method", "retrieval_type"]
    )
    by_category = metric_summary(rows, ["model", "method", "minerva_category"]).sort_values(
        ["model", "method", "minerva_category"]
    )

    overall.to_csv(out_dir / "overall_by_condition.csv", index=False)
    by_model.to_csv(out_dir / "overall_by_model.csv", index=False)
    by_method.to_csv(out_dir / "overall_by_method.csv", index=False)
    by_book.to_csv(out_dir / "overall_by_book.csv", index=False)
    by_book_method.to_csv(out_dir / "overall_by_book_method.csv", index=False)
    by_model_method.to_csv(out_dir / "overall_by_model_method.csv", index=False)
    by_retrieval.to_csv(out_dir / "overall_by_retrieval_model_method.csv", index=False)
    by_category.to_csv(out_dir / "overall_by_category_model_method.csv", index=False)

    # Model ordering from performance
    model_order = by_model["model"].tolist()

    # Extra model-level and category visualizations
    barplot(
        df=by_model,
        x="model",
        y="mean_f1",
        hue=None,
        title="Overall Mean F1 by Model",
        out_path=out_dir / "bar_mean_f1_by_model.png",
        order=model_order,
        rotate_xticks=15,
    )
    barplot(
        df=by_model_method,
        x="model",
        y="mean_f1",
        hue="method",
        title="Mean F1 by Model and Method",
        out_path=out_dir / "bar_mean_f1_by_model_method.png",
        order=model_order,
        hue_order=METHOD_ORDER,
        rotate_xticks=15,
    )
    for method in METHOD_ORDER:
        lineplot_category_by_method(
            by_category=by_category,
            method=method,
            model_order=model_order,
            out_path=out_dir / f"line_category_f1_by_model_{method}.png",
        )

    # Forest plot for model mean F1 with bootstrap CIs
    f1_ci_rows: list[dict[str, object]] = []
    for i, model in enumerate(model_order):
        g = rows[rows["model"] == model]
        mean_, lo, hi, _ = mean_ci(g["f1"], n_boot=args.bootstrap, seed=args.seed + 200 + i)
        f1_ci_rows.append({"model": model, "n": len(g), "mean_f1": mean_, "ci_low": lo, "ci_high": hi})
    model_f1_ci = pd.DataFrame(f1_ci_rows)
    model_f1_ci.to_csv(out_dir / "model_f1_with_ci.csv", index=False)
    forest_with_ci(
        table=model_f1_ci,
        label_col="model",
        mean_col="mean_f1",
        lo_col="ci_low",
        hi_col="ci_high",
        title="Forest Plot: Mean F1 by Model (95% bootstrap CI)",
        out_path=out_dir / "forest_mean_f1_by_model.png",
        x_label="Mean F1",
        xlim=(0.0, 1.0),
    )

    # Chronological-awareness score tables + plots
    chrono_model_method, chrono_model = chronological_scores(rows)
    chrono_model_method = chrono_model_method.sort_values(["model", "method"])
    chrono_model = chrono_model.sort_values("model")
    chrono_model_method.to_csv(out_dir / "chronological_awareness_by_model_method.csv", index=False)
    chrono_model.to_csv(out_dir / "chronological_awareness_by_model.csv", index=False)

    barplot(
        df=chrono_model,
        x="model",
        y="chronological_awareness_score",
        hue=None,
        title="Chronological Awareness Score by Model",
        out_path=out_dir / "bar_chronological_awareness_by_model.png",
        order=model_order,
        rotate_xticks=15,
    )
    barplot(
        df=chrono_model_method,
        x="model",
        y="chronological_awareness_score",
        hue="method",
        title="Chronological Awareness by Model and Method",
        out_path=out_dir / "bar_chronological_awareness_by_model_method.png",
        order=model_order,
        hue_order=METHOD_ORDER,
        rotate_xticks=15,
    )

    # Forest plot for chronological-awareness with bootstrap CI
    chrono_ci_rows: list[dict[str, object]] = []
    for i, model in enumerate(model_order):
        g = rows[rows["model"] == model].copy()
        if g.empty:
            continue
        rng = np.random.default_rng(args.seed + 500 + i)
        n = len(g)
        boot_vals: list[float] = []
        for _ in range(args.bootstrap):
            sample_idx = rng.integers(0, n, size=n)
            s = g.iloc[sample_idx]
            awareness = _chrono_components(s)[2]
            if not math.isnan(awareness):
                boot_vals.append(float(awareness))
        awareness = _chrono_components(g)[2]
        if boot_vals:
            lo, hi = np.percentile(np.asarray(boot_vals), [2.5, 97.5])
            lo_f, hi_f = float(lo), float(hi)
        else:
            lo_f, hi_f = float("nan"), float("nan")
        chrono_ci_rows.append(
            {
                "model": model,
                "n": n,
                "chronological_awareness_score": awareness,
                "ci_low": lo_f,
                "ci_high": hi_f,
            }
        )
    chrono_ci = pd.DataFrame(chrono_ci_rows)
    chrono_ci.to_csv(out_dir / "chronological_awareness_with_ci_by_model.csv", index=False)
    forest_with_ci(
        table=chrono_ci.dropna(subset=["chronological_awareness_score", "ci_low", "ci_high"]),
        label_col="model",
        mean_col="chronological_awareness_score",
        lo_col="ci_low",
        hi_col="ci_high",
        title="Forest Plot: Chronological Awareness by Model (95% bootstrap CI)",
        out_path=out_dir / "forest_chronological_awareness_by_model.png",
        x_label="Chronological Awareness Score",
        xlim=(0.0, 1.0),
    )

    # Heatmaps
    f1_pivot = overall.pivot(index="model", columns="condition", values="mean_f1")
    f1_pivot = f1_pivot.reindex(index=model_order, columns=CONDITION_ORDER)
    heatmap(
        f1_pivot,
        title="Mean F1 Heatmap: Model x Book-Method",
        out_path=out_dir / "heatmap_mean_f1_model_condition.png",
        cmap="YlGnBu",
    )

    em_pivot = overall.pivot(index="model", columns="condition", values="exact_match_rate")
    em_pivot = em_pivot.reindex(index=model_order, columns=CONDITION_ORDER)
    heatmap(
        em_pivot,
        title="Exact Match Heatmap: Model x Book-Method",
        out_path=out_dir / "heatmap_exact_match_model_condition.png",
        cmap="YlOrBr",
    )

    # DP-RAG deltas by model x book
    delta_model_book = (
        overall.pivot_table(index=["model", "book"], columns="method", values="mean_f1")
        .reset_index()
        .assign(diff_dp_minus_rag=lambda d: d["DP"] - d["RAG"])
    )
    delta_pivot = delta_model_book.pivot(index="model", columns="book", values="diff_dp_minus_rag")
    delta_pivot = delta_pivot.reindex(index=model_order, columns=BOOK_ORDER)
    heatmap(
        delta_pivot,
        title="F1 Delta Heatmap (DP - RAG): Model x Book",
        out_path=out_dir / "heatmap_delta_dp_minus_rag_model_book.png",
        cmap="RdBu_r",
        center=0.0,
    )

    # Retrieval-type weighted delta heatmap (RAG - DP)
    retrieval_delta = retrieval_delta_table(rows)
    retrieval_delta.to_csv(out_dir / "retrieval_type_delta_rag_minus_dp_by_model.csv", index=False)
    retrieval_order = (
        retrieval_delta.groupby("retrieval_type", as_index=False)["delta_rag_minus_dp"]
        .mean()
        .sort_values("delta_rag_minus_dp", ascending=False)["retrieval_type"]
        .tolist()
    )
    retrieval_delta_pivot = retrieval_delta.pivot(
        index="model", columns="retrieval_type", values="delta_rag_minus_dp"
    )
    retrieval_delta_pivot = retrieval_delta_pivot.reindex(index=model_order, columns=retrieval_order)
    heatmap(
        retrieval_delta_pivot,
        title="Weighted Mean F1 Delta (RAG - DP) by Retrieval Type",
        out_path=out_dir / "heatmap_retrieval_delta_rag_minus_dp.png",
        cmap="RdBu_r",
        center=0.0,
    )

    # Category heatmaps by method
    for method in METHOD_ORDER:
        cat = rows[rows["method"] == method].copy()
        if cat.empty:
            continue
        cat_tbl = (
            cat.groupby(["model", "minerva_category"], as_index=False)["f1"]
            .mean()
            .pivot(index="model", columns="minerva_category", values="f1")
        )
        cat_tbl = cat_tbl.reindex(index=model_order, columns=CATEGORY_ORDER)
        heatmap(
            cat_tbl,
            title=f"Category Mean F1 Heatmap ({method})",
            out_path=out_dir / f"heatmap_category_f1_{method}.png",
            cmap="PuBuGn",
        )

    # Paired DP vs RAG analysis
    pairs = paired_differences(rows)
    missing_notes: list[str] = []
    if not pairs.empty:
        pairs.to_csv(out_dir / "paired_row_differences_dp_minus_rag.csv", index=False)
        pairs = pairs.copy()
        pairs["delta_f1_rag_minus_dp"] = pairs["f1_rag"] - pairs["f1_dp"]
        pairs.to_csv(out_dir / "paired_row_differences_rag_minus_dp.csv", index=False)

        # Win/Tie/Lose summaries from matched row_key pairs
        wt_overall, wt_by_model, wt_by_model_book = pair_outcomes(pairs)
        wt_overall.to_csv(out_dir / "win_tie_loss_overall.csv", index=False)
        wt_by_model.to_csv(out_dir / "win_tie_loss_by_model.csv", index=False)
        wt_by_model_book.to_csv(out_dir / "win_tie_loss_by_model_book.csv", index=False)

        stacked_percent_bar_from_counts(
            counts=wt_by_model,
            index_cols=["model"],
            category_col="pair_outcome",
            order=model_order,
            category_order=["RAG wins", "Tie", "RAG loses"],
            title="Matched Row Pairs: RAG Wins vs Ties vs Losses by Model",
            out_path=out_dir / "stacked_win_tie_loss_by_model.png",
        )

        by_book_counts = (
            wt_by_model_book.groupby(["book", "pair_outcome"], as_index=False)
            .agg(n=("n", "sum"))
        )
        by_book_counts["pct"] = by_book_counts.groupby("book")["n"].transform(lambda s: 100.0 * s / s.sum())
        stacked_percent_bar_from_counts(
            counts=by_book_counts,
            index_cols=["book"],
            category_col="pair_outcome",
            order=BOOK_ORDER,
            category_order=["RAG wins", "Tie", "RAG loses"],
            title="Matched Row Pairs: RAG Wins vs Ties vs Losses by Book",
            out_path=out_dir / "stacked_win_tie_loss_by_book.png",
        )

        # ECDF of per-question delta F1 (RAG - DP)
        ecdf_delta_plot(
            pairs=pairs,
            model_order=model_order,
            out_path=out_dir / "ecdf_delta_f1_rag_minus_dp_by_model.png",
        )

        diff_rows = []
        for model, g in pairs.groupby("model"):
            mean_, lo, hi, _ = mean_ci(g["diff_f1"], args.bootstrap, args.seed + 10)
            diff_rows.append({"model": model, "n": len(g), "mean_diff_f1": mean_, "ci_low": lo, "ci_high": hi})
        diffs_model = pd.DataFrame(diff_rows).sort_values("mean_diff_f1", ascending=False)
        diffs_model.to_csv(out_dir / "dp_minus_rag_effect_by_model.csv", index=False)

        diff_rows = []
        for book, g in pairs.groupby("book"):
            mean_, lo, hi, _ = mean_ci(g["diff_f1"], args.bootstrap, args.seed + 20)
            diff_rows.append({"book": book, "n": len(g), "mean_diff_f1": mean_, "ci_low": lo, "ci_high": hi})
        diffs_book = pd.DataFrame(diff_rows).sort_values("mean_diff_f1", ascending=False)
        diffs_book.to_csv(out_dir / "dp_minus_rag_effect_by_book.csv", index=False)

        # Forest summary: one point per model-dataset (book) with bootstrap CI on mean delta (RAG - DP)
        effect_rows = []
        for i, ((model, book), g) in enumerate(pairs.groupby(["model", "book"], dropna=False)):
            mean_, lo, hi, _ = mean_ci(g["delta_f1_rag_minus_dp"], args.bootstrap, args.seed + 800 + i)
            effect_rows.append(
                {
                    "model": model,
                    "book": book,
                    "label": f"{model} | {book}",
                    "n": len(g),
                    "mean_delta_f1_rag_minus_dp": mean_,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
        effect_tbl = pd.DataFrame(effect_rows).sort_values("mean_delta_f1_rag_minus_dp", ascending=False)
        effect_tbl.to_csv(out_dir / "forest_effect_by_model_book_rag_minus_dp.csv", index=False)
        forest_with_ci(
            table=effect_tbl.rename(columns={"mean_delta_f1_rag_minus_dp": "mean"}),
            label_col="label",
            mean_col="mean",
            lo_col="ci_low",
            hi_col="ci_high",
            title="Forest Plot: Mean Delta F1 (RAG - DP) by Model-Book",
            out_path=out_dir / "forest_delta_f1_rag_minus_dp_by_model_book.png",
            x_label="Mean Delta F1 (RAG - DP)",
        )

        gardner_altman_by_group(
            pairs=pairs,
            group_col="model",
            out_path=out_dir / "gardner_altman_dp_minus_rag_by_model.png",
            title="Gardner-Altman Style Estimation: DP vs RAG by Model",
            n_boot=args.bootstrap,
            seed=args.seed,
        )
        gardner_altman_by_group(
            pairs=pairs,
            group_col="book",
            out_path=out_dir / "gardner_altman_dp_minus_rag_by_book.png",
            title="Gardner-Altman Style Estimation: DP vs RAG by Book",
            n_boot=args.bootstrap,
            seed=args.seed,
        )
    else:
        diffs_model = pd.DataFrame()
        diffs_book = pd.DataFrame()
        missing_notes.append("No paired DP/RAG rows could be matched using row_key.")

    # Error-profile stacked bars
    stacked_error_profile(rows=rows, out_dir=out_dir, model_order=model_order)

    # Data-quality notes
    counts = overall[["model", "condition", "n_rows"]].copy()
    if not counts["n_rows"].nunique() == 1:
        for _, r in counts[counts["n_rows"] != counts["n_rows"].max()].iterrows():
            missing_notes.append(
                f"{r['model']} / {r['condition']} has {int(r['n_rows'])} rows (expected {int(counts['n_rows'].max())})."
            )

    write_report(
        out_dir=out_dir,
        overall=overall,
        by_model=by_model,
        by_method=by_method,
        by_book_method=by_book_method,
        by_category=by_category,
        diffs_model=diffs_model,
        diffs_book=diffs_book,
        missing_notes=missing_notes,
    )

    print(f"Done. Wrote analysis to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
