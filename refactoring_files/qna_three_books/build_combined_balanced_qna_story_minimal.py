#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 150-question datasets per book with exact tuple balance and minimal "
            "story-grounded synthetic additions for missing Minerva-like categories."
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
        "--output-dir",
        type=Path,
        default=Path("refactoring_files") / "qna_three_books" / "combined_balanced_150_story_minimal",
    )
    parser.add_argument("--total-questions", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-edit-count", type=int, default=5)
    parser.add_argument("--spot-diff-count", type=int, default=5)
    return parser.parse_args()


def parse_answer_set(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({str(x) for x in value})
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return sorted({str(x) for x in parsed})
        if isinstance(parsed, dict):
            return sorted({str(k) for k in parsed.keys()})
        return [str(parsed)]
    except Exception:
        return [text]


def pick_base_tuple_balanced(df: pd.DataFrame, total_questions: int, seed: int) -> pd.DataFrame:
    cues = sorted(df["cue"].dropna().unique().tolist())
    if len(cues) == 0:
        raise ValueError("No cues found.")
    if total_questions % len(cues) != 0:
        raise ValueError(f"total_questions={total_questions} is not divisible by #cues={len(cues)}")

    per_cue = total_questions // len(cues)
    parts: list[pd.DataFrame] = []

    for i, cue in enumerate(cues):
        g = df[df["cue"] == cue].copy()
        g = g.sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
        g["__row_id"] = range(len(g))

        if cue == "(*, *, ent, *)":
            g_all = g[g["get"] == "all"].head(4)
            g_latest = g[g["get"] == "latest"].head(3)
            g_chrono = g[g["get"] == "chronological"].head(3)
            chosen = pd.concat([g_all, g_latest, g_chrono], ignore_index=True)
            if len(chosen) < per_cue:
                extra = g[~g["__row_id"].isin(chosen["__row_id"])].head(per_cue - len(chosen))
                chosen = pd.concat([chosen, extra], ignore_index=True)
            chosen = chosen.head(per_cue)
        elif cue == "(t, s, ent, c)":
            g_oe = g[g["retrieval_type"] == "Other entities"].head(5)
            g_fed = g[g["retrieval_type"] == "Full event details"].head(5)
            chosen = pd.concat([g_oe, g_fed], ignore_index=True)
            if len(chosen) < per_cue:
                extra = g[~g["__row_id"].isin(chosen["__row_id"])].head(per_cue - len(chosen))
                chosen = pd.concat([chosen, extra], ignore_index=True)
            chosen = chosen.head(per_cue)
        else:
            chosen = g.head(per_cue).copy()

        if len(chosen) != per_cue:
            raise ValueError(f"Cue {cue}: expected {per_cue} rows, got {len(chosen)}")

        parts.append(chosen)

    out = pd.concat(parts, ignore_index=True)
    if "__row_id" in out.columns:
        out = out.drop(columns=["__row_id"])
    out = out.sample(frac=1.0, random_state=seed + 999).reset_index(drop=True)
    return out


def make_recall_edit_row(donor: pd.Series, remove_item: str) -> pd.Series:
    ans = parse_answer_set(donor.get("correct_answer"))
    remaining = [x for x in ans if x != remove_item]

    row = donor.copy()
    row["question"] = (
        "Story recall-and-edit: Start from the answer to this story query: "
        f"\"{donor.get('question', '')}\". Remove \"{remove_item}\" from that answer. "
        "Return only the remaining items."
    )
    row["correct_answer"] = str(set(remaining))
    row["correct_answer_detailed"] = str({"remaining_items": remaining})
    row["correct_answer_chapters"] = ""
    row["retrieval_type"] = "story_edit::remove_one"
    row["get"] = "all"
    row["minerva_category"] = "Recall and Edit"
    row["minerva_subtask"] = "Story-grounded remove-one edit"
    row["combined_source"] = "story_synthetic"
    row["is_synthetic"] = True
    row["synthetic_type"] = "recall_and_edit"
    row["synthetic_note"] = f"Removed item: {remove_item}"
    return row


def make_spot_diff_row(donor_a: pd.Series, donor_b: pd.Series) -> pd.Series:
    a_items = parse_answer_set(donor_a.get("correct_answer"))
    b_items = parse_answer_set(donor_b.get("correct_answer"))
    set_a = set(a_items)
    set_b = set(b_items)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    ref = {"only_in_A": only_a, "only_in_B": only_b}

    row = donor_a.copy()
    row["question"] = (
        "Story spot-the-difference: Compare the answers to these two story queries and list differences.\n"
        f"A) {donor_a.get('question', '')}\n"
        f"B) {donor_b.get('question', '')}\n"
        "Return two lists: only_in_A and only_in_B."
    )
    row["correct_answer"] = json.dumps(ref, ensure_ascii=False)
    row["correct_answer_detailed"] = json.dumps(ref, ensure_ascii=False)
    row["correct_answer_chapters"] = ""
    row["retrieval_type"] = "story_diff::set_difference"
    row["get"] = "all"
    row["minerva_category"] = "Spot the Differences"
    row["minerva_subtask"] = "Story-grounded set difference"
    row["combined_source"] = "story_synthetic"
    row["is_synthetic"] = True
    row["synthetic_type"] = "spot_the_differences"
    row["synthetic_note"] = f"Compared q_idx {donor_a.get('q_idx')} vs {donor_b.get('q_idx')}"
    return row


def ensure_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "combined_source" not in out.columns:
        out["combined_source"] = "original"
    if "is_synthetic" not in out.columns:
        out["is_synthetic"] = False
    if "synthetic_type" not in out.columns:
        out["synthetic_type"] = ""
    if "synthetic_note" not in out.columns:
        out["synthetic_note"] = ""
    return out


def inject_minimal_story_synthetics(
    df: pd.DataFrame, recall_edit_count: int, spot_diff_count: int, seed: int
) -> pd.DataFrame:
    out = ensure_extra_columns(df)

    # Re-label an existing story-native subset as Composite (no synthetic needed).
    mask_composite = (
        (out["cue"] == "(t, s, ent, c)")
        & (out["retrieval_type"] == "Full event details")
    )
    out.loc[mask_composite, "minerva_category"] = "Composite"
    out.loc[mask_composite, "minerva_subtask"] = "Story full event details"

    # Donor pool: prefer Search rows with set-like answers, so tuple balance stays unchanged after replacement.
    search_rows = out[out["minerva_category"] == "Search"].copy()
    search_rows = (
        search_rows.sample(frac=1.0, random_state=seed)
        .reset_index()
        .rename(columns={"index": "_row_pos"})
    )

    # Recall-and-edit replacements.
    re_done = 0
    for _, donor in search_rows.iterrows():
        if re_done >= recall_edit_count:
            break
        items = parse_answer_set(donor.get("correct_answer"))
        if len(items) < 2:
            continue
        remove_item = items[0]
        new_row = make_recall_edit_row(donor, remove_item)
        out.loc[int(donor["_row_pos"])] = new_row
        re_done += 1

    # Spot-the-difference replacements.
    # Build pairs from remaining Search rows with at least 1 item each.
    current_search = out[out["minerva_category"] == "Search"].copy()
    current_search = (
        current_search.sample(frac=1.0, random_state=seed + 1000)
        .reset_index()
        .rename(columns={"index": "_row_pos"})
    )
    valid = []
    for _, r in current_search.iterrows():
        if len(parse_answer_set(r.get("correct_answer"))) >= 1:
            valid.append(r)
    sd_done = 0
    i = 0
    while i + 1 < len(valid) and sd_done < spot_diff_count:
        a = valid[i]
        b = valid[i + 1]
        new_row = make_spot_diff_row(a, b)
        out.loc[int(a["_row_pos"])] = new_row
        sd_done += 1
        i += 2

    out = out.reset_index(drop=True)
    out["combined_q_idx"] = out.index
    return out


def summarize(df: pd.DataFrame, file_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_cat = (
        df["minerva_category"].value_counts().rename_axis("minerva_category").reset_index(name="count")
    )
    by_cat["share"] = by_cat["count"] / len(df)
    by_cat.insert(0, "file", file_label)

    by_tuple = df["cue"].value_counts().rename_axis("cue").reset_index(name="count")
    by_tuple["share"] = by_tuple["count"] / len(df)
    by_tuple.insert(0, "file", file_label)

    by_source = df["combined_source"].value_counts().rename_axis("combined_source").reset_index(name="count")
    by_source["share"] = by_source["count"] / len(df)
    by_source.insert(0, "file", file_label)
    return by_cat, by_tuple, by_source


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matching {args.input_glob} in {args.input_dir}")

    all_cat: list[pd.DataFrame] = []
    all_tuple: list[pd.DataFrame] = []
    all_source: list[pd.DataFrame] = []

    for i, p in enumerate(files):
        df = pd.read_csv(p)
        base = pick_base_tuple_balanced(df, args.total_questions, args.seed + i * 101)
        combined = inject_minimal_story_synthetics(
            base,
            recall_edit_count=args.recall_edit_count,
            spot_diff_count=args.spot_diff_count,
            seed=args.seed + i * 101 + 7,
        )

        out_name = p.name.replace(".minerva_categorized.csv", ".combined_story_minimal_150.csv")
        out_path = args.output_dir / out_name
        combined.to_csv(out_path, index=False)

        c1, c2, c3 = summarize(combined, out_name)
        all_cat.append(c1)
        all_tuple.append(c2)
        all_source.append(c3)

    by_file_cat = pd.concat(all_cat, ignore_index=True)
    by_file_tuple = pd.concat(all_tuple, ignore_index=True)
    by_file_source = pd.concat(all_source, ignore_index=True)

    overall_cat = by_file_cat.groupby("minerva_category", as_index=False)["count"].sum()
    overall_cat["share"] = overall_cat["count"] / overall_cat["count"].sum()
    overall_cat = overall_cat.sort_values("count", ascending=False)

    overall_tuple = by_file_tuple.groupby("cue", as_index=False)["count"].sum()
    overall_tuple["share"] = overall_tuple["count"] / overall_tuple["count"].sum()
    overall_tuple = overall_tuple.sort_values("count", ascending=False)

    overall_source = by_file_source.groupby("combined_source", as_index=False)["count"].sum()
    overall_source["share"] = overall_source["count"] / overall_source["count"].sum()
    overall_source = overall_source.sort_values("count", ascending=False)

    by_file_cat.to_csv(args.output_dir / "summary_by_file_minerva_category.csv", index=False)
    by_file_tuple.to_csv(args.output_dir / "summary_by_file_tuple.csv", index=False)
    by_file_source.to_csv(args.output_dir / "summary_by_file_source.csv", index=False)
    overall_cat.to_csv(args.output_dir / "summary_overall_minerva_category.csv", index=False)
    overall_tuple.to_csv(args.output_dir / "summary_overall_tuple.csv", index=False)
    overall_source.to_csv(args.output_dir / "summary_overall_source.csv", index=False)

    print(f"Built {len(files)} files in {args.output_dir}")
    print(
        "Synthetic additions per file:",
        f"Recall and Edit={args.recall_edit_count}, Spot the Differences={args.spot_diff_count}",
    )


if __name__ == "__main__":
    main()
