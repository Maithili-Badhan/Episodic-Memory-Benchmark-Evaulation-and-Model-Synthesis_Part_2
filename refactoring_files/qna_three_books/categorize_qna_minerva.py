#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MINERVA_CATEGORIES = [
    "Search",
    "Recall and Edit",
    "Match and Compare",
    "Spot the Differences",
    "Compute on Sets and Lists",
    "Stateful Processing",
    "Composite",
]

AXES = ["t", "s", "ent", "c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Categorize Episodic Memory QA rows into Minerva-style categories and "
            "produce per-file analysis summaries."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("refactoring_files") / "qna_three_books",
        help="Directory containing the QA CSV files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_df_qa.csv",
        help="Glob pattern for selecting input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("refactoring_files") / "qna_three_books" / "minerva_analysis",
        help="Where categorized files and summaries will be written.",
    )
    return parser.parse_args()


def parse_cue_tuple(cue: str) -> list[str]:
    if not isinstance(cue, str):
        return []
    text = cue.strip()
    if not text.startswith("(") or not text.endswith(")"):
        return []
    parts = [p.strip() for p in text[1:-1].split(",")]
    return parts


def cue_missing_elements(cue: str) -> str:
    parts = parse_cue_tuple(cue)
    if len(parts) != 4:
        return ""
    missing = [AXES[i] for i, token in enumerate(parts) if token == "*"]
    return "|".join(missing)


def cue_present_elements(cue: str) -> str:
    parts = parse_cue_tuple(cue)
    if len(parts) != 4:
        return ""
    present = [AXES[i] for i, token in enumerate(parts) if token != "*"]
    return "|".join(present)


def map_to_minerva_category(row: pd.Series) -> tuple[str, str]:
    get_style = str(row.get("get", "")).strip().lower()
    retrieval_type = str(row.get("retrieval_type", "")).strip()

    # Heuristic mapping from episodic QA behavior to Minerva category families.
    if get_style == "chronological":
        return (
            "Compute on Sets and Lists",
            "Temporal ordering over retrieved events",
        )
    if get_style == "latest":
        return (
            "Stateful Processing",
            "Latest-state retrieval in an event stream",
        )
    if retrieval_type == "Other entities":
        return (
            "Match and Compare",
            "Co-occurrence lookup excluding anchor entity",
        )
    return (
        "Search",
        "Keyed retrieval from episodic event memory",
    )


def summarize_categories(df: pd.DataFrame, file_label: str) -> pd.DataFrame:
    summary = (
        df.groupby(["minerva_category", "minerva_subtask"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    total = len(df)
    summary["share"] = summary["count"] / total
    summary.insert(0, "file", file_label)
    return summary


def summarize_missing_tuple_elements(df: pd.DataFrame, file_label: str) -> pd.DataFrame:
    summary = (
        df.groupby(["cue", "missing_tuple_elements", "present_tuple_elements"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    total = len(df)
    summary["share"] = summary["count"] / total
    summary.insert(0, "file", file_label)
    return summary


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{args.glob}' under {input_dir}")

    all_cat_summaries: list[pd.DataFrame] = []
    all_tuple_summaries: list[pd.DataFrame] = []
    all_rows: list[pd.DataFrame] = []

    for csv_path in files:
        df = pd.read_csv(csv_path)
        file_label = csv_path.name

        categories = df.apply(map_to_minerva_category, axis=1, result_type="expand")
        categories.columns = ["minerva_category", "minerva_subtask"]
        out_df = pd.concat([df, categories], axis=1)
        out_df["missing_tuple_elements"] = out_df["cue"].apply(cue_missing_elements)
        out_df["present_tuple_elements"] = out_df["cue"].apply(cue_present_elements)
        out_df["minerva_category_known"] = out_df["minerva_category"].isin(MINERVA_CATEGORIES)

        categorized_path = output_dir / f"{csv_path.stem}.minerva_categorized.csv"
        out_df.to_csv(categorized_path, index=False)

        cat_summary = summarize_categories(out_df, file_label)
        tuple_summary = summarize_missing_tuple_elements(out_df, file_label)
        all_cat_summaries.append(cat_summary)
        all_tuple_summaries.append(tuple_summary)
        all_rows.append(out_df.assign(source_file=file_label))

    by_file_cat = pd.concat(all_cat_summaries, ignore_index=True)
    by_file_tuple = pd.concat(all_tuple_summaries, ignore_index=True)
    all_df = pd.concat(all_rows, ignore_index=True)

    overall_cat = (
        all_df.groupby(["minerva_category", "minerva_subtask"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    overall_cat["share"] = overall_cat["count"] / len(all_df)

    overall_tuple = (
        all_df.groupby(["cue", "missing_tuple_elements", "present_tuple_elements"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    overall_tuple["share"] = overall_tuple["count"] / len(all_df)

    by_file_cat.to_csv(output_dir / "summary_by_file_minerva_category.csv", index=False)
    overall_cat.to_csv(output_dir / "summary_overall_minerva_category.csv", index=False)
    by_file_tuple.to_csv(output_dir / "summary_by_file_tuple_missingness.csv", index=False)
    overall_tuple.to_csv(output_dir / "summary_overall_tuple_missingness.csv", index=False)

    print(f"Processed {len(files)} files. Output dir: {output_dir}")
    print("Per-file category summary:")
    print(by_file_cat.to_string(index=False))
    print("\nOverall category summary:")
    print(overall_cat.to_string(index=False))


if __name__ == "__main__":
    main()
