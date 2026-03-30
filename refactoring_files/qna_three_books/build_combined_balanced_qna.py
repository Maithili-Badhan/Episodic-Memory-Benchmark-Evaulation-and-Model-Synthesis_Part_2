#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

SNAPSHOT_DIR_BY_CATEGORY = {
    "Search": "search",
    "Recall and Edit": "recall_and_edit",
    "Match and Compare": "match_and_compare",
    "Spot the Differences": "spot_the_differences",
    "Compute on Sets and Lists": "compute_on_sets_and_lists",
    "Stateful Processing": "stateful_processing",
    "Composite": "composite",
}


@dataclass
class SnapshotItem:
    category: str
    task: str
    sid: str
    prompt: str
    reference: object
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one 150-QnA combined dataset per book with exact tuple balance and "
            "balanced Minerva categories using snapshot-based synthetic fill."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("refactoring_files") / "qna_three_books" / "minerva_analysis",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*_df_qa.minerva_categorized.csv",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path("refactoring_files") / "minerva" / "minerva_snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("refactoring_files") / "qna_three_books" / "combined_balanced_150",
    )
    parser.add_argument("--total-questions", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def category_targets_equal(total: int) -> dict[str, int]:
    base = total // len(MINERVA_CATEGORIES)
    remainder = total % len(MINERVA_CATEGORIES)
    targets: dict[str, int] = {}
    for i, cat in enumerate(MINERVA_CATEGORIES):
        targets[cat] = base + (1 if i < remainder else 0)
    return targets


def load_snapshot_pool(snapshot_root: Path) -> dict[str, list[SnapshotItem]]:
    pool: dict[str, list[SnapshotItem]] = {k: [] for k in MINERVA_CATEGORIES}
    for cat in MINERVA_CATEGORIES:
        sub = snapshot_root / SNAPSHOT_DIR_BY_CATEGORY[cat]
        if not sub.is_dir():
            continue
        for jsonl_path in sorted(sub.glob("*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    pool[cat].append(
                        SnapshotItem(
                            category=cat,
                            task=str(obj.get("task", "")),
                            sid=str(obj.get("id", "")),
                            prompt=str(obj.get("prompt", "")),
                            reference=obj.get("reference", ""),
                            source_file=str(jsonl_path),
                        )
                    )
    return pool


def pick_tuple_balanced_base(df: pd.DataFrame, total_questions: int, seed: int) -> pd.DataFrame:
    cues = sorted(df["cue"].dropna().unique().tolist())
    if not cues:
        raise ValueError("No cue values found; cannot enforce tuple balance.")
    if total_questions % len(cues) != 0:
        raise ValueError(
            f"total_questions={total_questions} not divisible by number of tuples={len(cues)}"
        )
    per_tuple = total_questions // len(cues)
    targets = category_targets_equal(total_questions)
    remaining = targets.copy()

    selected_parts: list[pd.DataFrame] = []
    for i, cue in enumerate(cues):
        g = df[df["cue"] == cue].copy()
        if len(g) < per_tuple:
            raise ValueError(f"Tuple {cue} has only {len(g)} rows, needs {per_tuple}")
        g = g.sample(frac=1.0, random_state=seed + i).reset_index(drop=False)
        g["cat_score"] = g["minerva_category"].map(lambda c: remaining.get(str(c), 0))
        g = g.sort_values(["cat_score", "index"], ascending=[False, True]).head(per_tuple)
        for cat in g["minerva_category"]:
            if cat in remaining:
                remaining[cat] -= 1
        selected_parts.append(g.drop(columns=["cat_score"]))

    out = pd.concat(selected_parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed + 999).reset_index(drop=True)
    return out


def make_synthetic_row(donor: pd.Series, item: SnapshotItem, new_category: str) -> pd.Series:
    row = donor.copy()
    row["question"] = (
        f"[Minerva/{item.category}/{item.task}] "
        f"{item.prompt}"
    )
    row["correct_answer"] = json.dumps(item.reference, ensure_ascii=False)
    row["correct_answer_detailed"] = json.dumps(item.reference, ensure_ascii=False)
    row["correct_answer_chapters"] = ""
    row["retrieval_type"] = f"minerva::{item.task}"
    row["get"] = "all"
    row["minerva_category"] = new_category
    row["minerva_subtask"] = f"Synthetic from Minerva snapshot: {item.task}"
    row["combined_source"] = "minerva_snapshot"
    row["is_synthetic"] = True
    row["synthetic_task"] = item.task
    row["synthetic_snapshot_id"] = item.sid
    row["synthetic_snapshot_file"] = item.source_file
    row["synthetic_reference_raw"] = json.dumps(item.reference, ensure_ascii=False)
    return row


def rebalance_categories(
    selected: pd.DataFrame,
    snapshot_pool: dict[str, list[SnapshotItem]],
    total_questions: int,
) -> pd.DataFrame:
    out = selected.copy()
    if "combined_source" not in out.columns:
        out["combined_source"] = "original"
    if "is_synthetic" not in out.columns:
        out["is_synthetic"] = False
    for col in [
        "synthetic_task",
        "synthetic_snapshot_id",
        "synthetic_snapshot_file",
        "synthetic_reference_raw",
        "original_minerva_category",
    ]:
        if col not in out.columns:
            out[col] = ""

    targets = category_targets_equal(total_questions)
    current = out["minerva_category"].value_counts().to_dict()
    snapshot_cursor = {k: 0 for k in MINERVA_CATEGORIES}

    def pick_snapshot(cat: str) -> SnapshotItem:
        items = snapshot_pool.get(cat, [])
        if not items:
            raise ValueError(f"No snapshot items available for category: {cat}")
        idx = snapshot_cursor[cat] % len(items)
        snapshot_cursor[cat] += 1
        return items[idx]

    def donor_candidates() -> list[str]:
        counts = out["minerva_category"].value_counts().to_dict()
        surplus = {k: counts.get(k, 0) - targets.get(k, 0) for k in MINERVA_CATEGORIES}
        return [k for k, v in sorted(surplus.items(), key=lambda x: x[1], reverse=True) if v > 0]

    for cat in MINERVA_CATEGORIES:
        need = targets[cat] - current.get(cat, 0)
        for _ in range(max(0, need)):
            donors = donor_candidates()
            if not donors:
                raise ValueError("No donor category with surplus found; cannot rebalance.")
            donor_cat = donors[0]
            donor_idx = out.index[out["minerva_category"] == donor_cat][0]
            donor_row = out.loc[donor_idx].copy()
            snapshot_item = pick_snapshot(cat)
            new_row = make_synthetic_row(donor_row, snapshot_item, cat)
            new_row["original_minerva_category"] = str(donor_cat)
            out.loc[donor_idx] = new_row

            current[donor_cat] = current.get(donor_cat, 0) - 1
            current[cat] = current.get(cat, 0) + 1

    out = out.reset_index(drop=True)
    out["combined_q_idx"] = out.index
    return out


def summarize_file(df: pd.DataFrame, file_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat = (
        df["minerva_category"]
        .value_counts()
        .rename_axis("minerva_category")
        .reset_index(name="count")
    )
    cat["share"] = cat["count"] / len(df)
    cat.insert(0, "file", file_label)

    tup = (
        df["cue"]
        .value_counts()
        .rename_axis("cue")
        .reset_index(name="count")
    )
    tup["share"] = tup["count"] / len(df)
    tup.insert(0, "file", file_label)
    return cat, tup


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(args.input_dir.glob(args.input_glob))
    if not input_files:
        raise FileNotFoundError(
            f"No categorized CSV files found in {args.input_dir} matching {args.input_glob}"
        )

    snapshot_pool = load_snapshot_pool(args.snapshot_root)
    for cat in MINERVA_CATEGORIES:
        if len(snapshot_pool.get(cat, [])) == 0:
            raise ValueError(f"Snapshot pool empty for category '{cat}'")

    all_cat_summary: list[pd.DataFrame] = []
    all_tuple_summary: list[pd.DataFrame] = []

    for i, csv_path in enumerate(input_files):
        df = pd.read_csv(csv_path)
        base = pick_tuple_balanced_base(df, args.total_questions, args.seed + i * 101)
        combined = rebalance_categories(base, snapshot_pool, args.total_questions)

        out_name = csv_path.name.replace(".minerva_categorized.csv", ".combined_balanced_150.csv")
        out_path = args.output_dir / out_name
        combined.to_csv(out_path, index=False)

        cat_sum, tuple_sum = summarize_file(combined, out_name)
        all_cat_summary.append(cat_sum)
        all_tuple_summary.append(tuple_sum)

    by_file_cat = pd.concat(all_cat_summary, ignore_index=True)
    by_file_tuple = pd.concat(all_tuple_summary, ignore_index=True)
    overall_cat = (
        by_file_cat.groupby("minerva_category", as_index=False)["count"].sum().sort_values(
            "count", ascending=False
        )
    )
    overall_cat["share"] = overall_cat["count"] / overall_cat["count"].sum()
    overall_tuple = (
        by_file_tuple.groupby("cue", as_index=False)["count"].sum().sort_values(
            "count", ascending=False
        )
    )
    overall_tuple["share"] = overall_tuple["count"] / overall_tuple["count"].sum()

    by_file_cat.to_csv(args.output_dir / "summary_by_file_minerva_category_combined.csv", index=False)
    overall_cat.to_csv(args.output_dir / "summary_overall_minerva_category_combined.csv", index=False)
    by_file_tuple.to_csv(args.output_dir / "summary_by_file_tuple_combined.csv", index=False)
    overall_tuple.to_csv(args.output_dir / "summary_overall_tuple_combined.csv", index=False)

    print(f"Built {len(input_files)} combined datasets in: {args.output_dir}")
    print("Category target used:", category_targets_equal(args.total_questions))


if __name__ == "__main__":
    main()
