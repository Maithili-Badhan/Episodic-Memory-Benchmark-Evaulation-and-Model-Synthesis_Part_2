#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATASET_GLOB = "*combined_story_minerva_balanced_150.csv"


def dataset_label(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("news_"):
        return "NEWS"
    if name.startswith("scifi_"):
        return "SCIFI"
    if name.startswith("thriller_"):
        return "THRILLER"
    return path.stem


def load_coverage() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(ROOT.glob(DATASET_GLOB)):
        df = pd.read_csv(csv_path)
        if "minerva_category" not in df.columns:
            continue

        lbl = dataset_label(csv_path)
        counts = (
            df["minerva_category"]
            .fillna("Unknown")
            .value_counts(dropna=False)
            .rename_axis("minerva_category")
            .reset_index(name="count")
        )
        counts["dataset"] = lbl
        counts["total_rows"] = len(df)
        counts["coverage_pct"] = (counts["count"] / len(df) * 100.0).round(2)
        frames.append(counts)

    if not frames:
        raise SystemExit("No compatible datasets found for coverage plotting.")

    out = pd.concat(frames, ignore_index=True)
    return out[["dataset", "minerva_category", "count", "total_rows", "coverage_pct"]]


def plot_grouped_counts(coverage: pd.DataFrame) -> None:
    order = sorted(coverage["minerva_category"].unique())
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=coverage,
        x="minerva_category",
        y="count",
        hue="dataset",
        order=order,
        palette="Set2",
    )
    plt.title("Minerva Category Coverage by Dataset (Counts)")
    plt.xlabel("Minerva Category")
    plt.ylabel("Question Count")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "minerva_category_counts_grouped.png", dpi=220)
    plt.close()


def plot_grouped_percentages(coverage: pd.DataFrame) -> None:
    order = sorted(coverage["minerva_category"].unique())
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=coverage,
        x="minerva_category",
        y="coverage_pct",
        hue="dataset",
        order=order,
        palette="Set2",
    )
    plt.title("Minerva Category Coverage by Dataset (Percentage)")
    plt.xlabel("Minerva Category")
    plt.ylabel("Coverage (%)")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "minerva_category_percent_grouped.png", dpi=220)
    plt.close()


def plot_heatmap(coverage: pd.DataFrame) -> None:
    pivot = coverage.pivot_table(
        index="dataset",
        columns="minerva_category",
        values="coverage_pct",
        aggfunc="sum",
        fill_value=0.0,
    )
    plt.figure(figsize=(11, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=100)
    plt.title("Minerva Category Coverage Heatmap (%)")
    plt.xlabel("Minerva Category")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "minerva_category_coverage_heatmap.png", dpi=220)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    coverage = load_coverage().sort_values(["dataset", "minerva_category"]).reset_index(drop=True)
    coverage.to_csv(OUT_DIR / "minerva_category_coverage_summary.csv", index=False)

    plot_grouped_counts(coverage)
    plot_grouped_percentages(coverage)
    plot_heatmap(coverage)

    print("Saved summary:", OUT_DIR / "minerva_category_coverage_summary.csv")
    print("Saved plot:", OUT_DIR / "minerva_category_counts_grouped.png")
    print("Saved plot:", OUT_DIR / "minerva_category_percent_grouped.png")
    print("Saved plot:", OUT_DIR / "minerva_category_coverage_heatmap.png")


if __name__ == "__main__":
    main()
