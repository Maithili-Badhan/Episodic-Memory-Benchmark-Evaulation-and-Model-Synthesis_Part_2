#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebalance Minerva categories for story-only 150-QnA files while keeping "
            "tuple distribution unchanged."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("refactoring_files")
        / "qna_three_books"
        / "combined_balanced_150_story_minimal",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*.combined_story_minimal_150.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("refactoring_files")
        / "qna_three_books"
        / "combined_balanced_150_story_minerva_balanced",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def category_targets_equal(total: int) -> dict[str, int]:
    base = total // len(MINERVA_CATEGORIES)
    rem = total % len(MINERVA_CATEGORIES)
    out: dict[str, int] = {}
    for i, cat in enumerate(MINERVA_CATEGORIES):
        out[cat] = base + (1 if i < rem else 0)
    return out


def parse_obj(v: Any) -> Any:
    if isinstance(v, (list, tuple, dict, set, int, float)):
        return v
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def parse_items(v: Any) -> list[str]:
    obj = parse_obj(v)
    if obj is None:
        return []
    if isinstance(obj, dict):
        # For dict-like answers, use values then keys as fallback.
        vals = [str(x) for x in obj.values() if str(x).strip()]
        if vals:
            return sorted(set(vals))
        return sorted(set(str(k) for k in obj.keys()))
    if isinstance(obj, set):
        return sorted(str(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return sorted(set(str(x) for x in obj))
    return [str(obj)]


def parse_chapters(v: Any) -> list[int]:
    obj = parse_obj(v)
    if obj is None:
        return []
    vals: list[Any]
    if isinstance(obj, dict):
        vals = list(obj.keys()) + list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        vals = list(obj)
    else:
        vals = [obj]
    out: list[int] = []
    for x in vals:
        try:
            out.append(int(x))
        except Exception:
            pass
    return sorted(set(out))


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "combined_source": "original",
        "is_synthetic": False,
        "synthetic_type": "",
        "synthetic_note": "",
        "original_minerva_category": "",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


def pick_partner(df: pd.DataFrame, donor_idx: int, rng: random.Random) -> pd.Series:
    donor = df.loc[donor_idx]
    same_cue = df[(df.index != donor_idx) & (df["cue"] == donor["cue"])]
    pool = same_cue if len(same_cue) > 0 else df[df.index != donor_idx]
    if len(pool) == 0:
        return donor
    return pool.iloc[rng.randrange(len(pool))]


def make_recall_edit(donor: pd.Series) -> tuple[str, Any, Any, str, str]:
    items = parse_items(donor.get("correct_answer"))
    if len(items) == 0:
        return (
            f"Story recall-and-edit: For query \"{donor.get('question','')}\", return an empty answer.",
            str(set()),
            str({"remaining_items": []}),
            "story_edit::empty",
            "all",
        )
    remove_item = items[0]
    remaining = [x for x in items if x != remove_item]
    q = (
        "Story recall-and-edit: Start from the answer to this story query: "
        f"\"{donor.get('question','')}\". Remove \"{remove_item}\" from the answer. "
        "Return only the remaining items."
    )
    a = str(set(remaining))
    d = str({"remaining_items": remaining})
    return q, a, d, "story_edit::remove_one", "all"


def make_match_compare(donor: pd.Series, partner: pd.Series) -> tuple[str, Any, Any, str, str]:
    a_items = set(parse_items(donor.get("correct_answer")))
    b_items = set(parse_items(partner.get("correct_answer")))
    inter = sorted(a_items.intersection(b_items))
    q = (
        "Story match-and-compare: Compare answers of these two story queries and list only shared items.\n"
        f"A) {donor.get('question','')}\n"
        f"B) {partner.get('question','')}\n"
        "Return the intersection."
    )
    a = str(set(inter))
    d = str({"shared_items": inter})
    return q, a, d, "story_compare::intersection", "all"


def make_spot_diff(donor: pd.Series, partner: pd.Series) -> tuple[str, Any, Any, str, str]:
    a_items = set(parse_items(donor.get("correct_answer")))
    b_items = set(parse_items(partner.get("correct_answer")))
    only_a = sorted(a_items - b_items)
    only_b = sorted(b_items - a_items)
    q = (
        "Story spot-the-differences: Compare answers of these two story queries.\n"
        f"A) {donor.get('question','')}\n"
        f"B) {partner.get('question','')}\n"
        "Return only_in_A and only_in_B."
    )
    ref = {"only_in_A": only_a, "only_in_B": only_b}
    a = json.dumps(ref, ensure_ascii=False)
    d = json.dumps(ref, ensure_ascii=False)
    return q, a, d, "story_diff::set_difference", "all"


def make_compute_lists(donor: pd.Series) -> tuple[str, Any, Any, str, str]:
    items = parse_items(donor.get("correct_answer"))
    ordered = sorted(items)
    q = (
        "Story compute-on-lists: For this story query answer, sort all returned items in ascending "
        f"alphabetical order.\nQuery: {donor.get('question','')}"
    )
    a = json.dumps(ordered, ensure_ascii=False)
    d = json.dumps({"sorted_items": ordered}, ensure_ascii=False)
    return q, a, d, "story_compute::sort_items", "chronological"


def make_stateful(donor: pd.Series) -> tuple[str, Any, Any, str, str]:
    chapters = parse_chapters(donor.get("correct_answer_chapters"))
    if len(chapters) == 0:
        # fallback from answer size
        latest = len(parse_items(donor.get("correct_answer")))
    else:
        latest = max(chapters)
    q = (
        "Story stateful-processing: For this story query, return only the latest chapter index "
        f"from supporting chapters.\nQuery: {donor.get('question','')}"
    )
    a = str(latest)
    d = str({"latest_chapter": latest})
    return q, a, d, "story_state::latest_chapter", "latest"


def make_composite(donor: pd.Series, partner: pd.Series) -> tuple[str, Any, Any, str, str]:
    a_items = set(parse_items(donor.get("correct_answer")))
    b_items = set(parse_items(partner.get("correct_answer")))
    out_items = sorted(a_items - b_items)
    q = (
        "Story composite task: Use two story queries.\n"
        f"A) {donor.get('question','')}\n"
        f"B) {partner.get('question','')}\n"
        "Take answer(A), remove all items that also appear in answer(B), then return the remaining "
        "items sorted alphabetically."
    )
    a = json.dumps(out_items, ensure_ascii=False)
    d = json.dumps({"remaining_sorted": out_items}, ensure_ascii=False)
    return q, a, d, "story_composite::difference_then_sort", "all"


def synthesize_for_category(
    target_cat: str, donor: pd.Series, partner: pd.Series
) -> tuple[str, Any, Any, str, str, str]:
    if target_cat == "Recall and Edit":
        q, a, d, rt, g = make_recall_edit(donor)
        return q, a, d, rt, g, "recall_and_edit"
    if target_cat == "Match and Compare":
        q, a, d, rt, g = make_match_compare(donor, partner)
        return q, a, d, rt, g, "match_and_compare"
    if target_cat == "Spot the Differences":
        q, a, d, rt, g = make_spot_diff(donor, partner)
        return q, a, d, rt, g, "spot_the_differences"
    if target_cat == "Compute on Sets and Lists":
        q, a, d, rt, g = make_compute_lists(donor)
        return q, a, d, rt, g, "compute_on_sets_and_lists"
    if target_cat == "Stateful Processing":
        q, a, d, rt, g = make_stateful(donor)
        return q, a, d, rt, g, "stateful_processing"
    if target_cat == "Composite":
        q, a, d, rt, g = make_composite(donor, partner)
        return q, a, d, rt, g, "composite"
    raise ValueError(f"Unsupported target category: {target_cat}")


def rebalance_file(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = ensure_cols(df)
    total = len(out)
    targets = category_targets_equal(total)
    rng = random.Random(seed)

    def counts() -> dict[str, int]:
        vc = out["minerva_category"].value_counts().to_dict()
        return {k: int(vc.get(k, 0)) for k in MINERVA_CATEGORIES}

    cur = counts()

    def deficits(cur_counts: dict[str, int]) -> dict[str, int]:
        return {k: targets[k] - cur_counts.get(k, 0) for k in MINERVA_CATEGORIES}

    def surpluses(cur_counts: dict[str, int]) -> dict[str, int]:
        return {k: cur_counts.get(k, 0) - targets[k] for k in MINERVA_CATEGORIES}

    while True:
        d = deficits(cur)
        needed = {k: v for k, v in d.items() if v > 0}
        if not needed:
            break

        s = surpluses(cur)
        donors = {k: v for k, v in s.items() if v > 0}
        if not donors:
            raise RuntimeError("No donor categories available while deficits remain.")

        # Pick largest deficit and largest surplus to converge quickly.
        target_cat = sorted(needed.items(), key=lambda x: (-x[1], x[0]))[0][0]
        donor_cat = sorted(donors.items(), key=lambda x: (-x[1], x[0]))[0][0]

        donor_candidates = out[out["minerva_category"] == donor_cat]
        if len(donor_candidates) == 0:
            raise RuntimeError(f"No donor rows for category {donor_cat}")
        donor_idx = donor_candidates.index[rng.randrange(len(donor_candidates))]
        donor = out.loc[donor_idx]
        partner = pick_partner(out, int(donor_idx), rng)

        q, a, det, rt, get_style, syn_type = synthesize_for_category(target_cat, donor, partner)
        out.at[donor_idx, "question"] = q
        out.at[donor_idx, "correct_answer"] = a
        out.at[donor_idx, "correct_answer_detailed"] = det
        out.at[donor_idx, "correct_answer_chapters"] = ""
        out.at[donor_idx, "retrieval_type"] = rt
        out.at[donor_idx, "get"] = get_style
        out.at[donor_idx, "original_minerva_category"] = donor_cat
        out.at[donor_idx, "minerva_category"] = target_cat
        out.at[donor_idx, "minerva_subtask"] = f"Story synthetic {syn_type}"
        out.at[donor_idx, "combined_source"] = "story_synthetic_balanced"
        out.at[donor_idx, "is_synthetic"] = True
        out.at[donor_idx, "synthetic_type"] = syn_type
        out.at[donor_idx, "synthetic_note"] = (
            f"Rebalanced from {donor_cat} to {target_cat}; cue preserved"
        )

        cur = counts()

    out = out.reset_index(drop=True)
    out["combined_q_idx"] = out.index
    return out


def summarize_file(df: pd.DataFrame, file_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_cat = (
        df["minerva_category"].value_counts().rename_axis("minerva_category").reset_index(name="count")
    )
    by_cat["share"] = by_cat["count"] / len(df)
    by_cat.insert(0, "file", file_label)

    by_tuple = df["cue"].value_counts().rename_axis("cue").reset_index(name="count")
    by_tuple["share"] = by_tuple["count"] / len(df)
    by_tuple.insert(0, "file", file_label)

    by_src = df["combined_source"].value_counts().rename_axis("combined_source").reset_index(name="count")
    by_src["share"] = by_src["count"] / len(df)
    by_src.insert(0, "file", file_label)
    return by_cat, by_tuple, by_src


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No input files in {args.input_dir} matching {args.input_glob}")

    all_cat: list[pd.DataFrame] = []
    all_tuple: list[pd.DataFrame] = []
    all_src: list[pd.DataFrame] = []

    for i, p in enumerate(files):
        df = pd.read_csv(p)
        out = rebalance_file(df, seed=args.seed + i * 101)
        out_name = p.name.replace(".combined_story_minimal_150.csv", ".combined_story_minerva_balanced_150.csv")
        out_path = args.output_dir / out_name
        out.to_csv(out_path, index=False)

        c1, c2, c3 = summarize_file(out, out_name)
        all_cat.append(c1)
        all_tuple.append(c2)
        all_src.append(c3)

    by_file_cat = pd.concat(all_cat, ignore_index=True)
    by_file_tuple = pd.concat(all_tuple, ignore_index=True)
    by_file_src = pd.concat(all_src, ignore_index=True)

    overall_cat = by_file_cat.groupby("minerva_category", as_index=False)["count"].sum()
    overall_cat["share"] = overall_cat["count"] / overall_cat["count"].sum()
    overall_cat = overall_cat.sort_values("count", ascending=False)

    overall_tuple = by_file_tuple.groupby("cue", as_index=False)["count"].sum()
    overall_tuple["share"] = overall_tuple["count"] / overall_tuple["count"].sum()
    overall_tuple = overall_tuple.sort_values("count", ascending=False)

    overall_src = by_file_src.groupby("combined_source", as_index=False)["count"].sum()
    overall_src["share"] = overall_src["count"] / overall_src["count"].sum()
    overall_src = overall_src.sort_values("count", ascending=False)

    by_file_cat.to_csv(args.output_dir / "summary_by_file_minerva_category.csv", index=False)
    by_file_tuple.to_csv(args.output_dir / "summary_by_file_tuple.csv", index=False)
    by_file_src.to_csv(args.output_dir / "summary_by_file_source.csv", index=False)
    overall_cat.to_csv(args.output_dir / "summary_overall_minerva_category.csv", index=False)
    overall_tuple.to_csv(args.output_dir / "summary_overall_tuple.csv", index=False)
    overall_src.to_csv(args.output_dir / "summary_overall_source.csv", index=False)

    print(f"Wrote balanced files to: {args.output_dir}")
    print("Target category counts per file:", category_targets_equal(150))


if __name__ == "__main__":
    main()
