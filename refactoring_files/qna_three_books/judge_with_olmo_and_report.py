#!/usr/bin/env python
"""
LLM judging (OLMo3 via OpenAI-compatible endpoint) + metrics + plots.

Typical flow:
1) Judge answers JSONL against ground-truth items (from answer rows or QA CSV).
2) Save judged JSONL.
3) Compute aggregate metrics/scores and generate CSV summaries + PNG plots.

Example:
  python refactoring_files/qna_three_books/judge_with_olmo_and_report.py ^
    --answers-jsonl refactoring_files/Models_answers/Qwen2.5_14B_instruct/news_qwen25_14b_dp.answers.jsonl ^
    --answers-jsonl refactoring_files/Models_answers/Qwen2.5_14B_instruct/scifi_qwen25_14b_dp.answers.jsonl ^
    --answers-jsonl refactoring_files/Models_answers/Qwen2.5_14B_instruct/thriller_qwen25_14b_dp.answers.jsonl ^
    --judge-model Olmo-3-7B-Instruct ^
    --base-url http://127.0.0.1:8080/v1 ^
    --output-dir refactoring_files/artifacts/judge_olmo
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


SYSTEM_PROMPT = (
    "You are a strict evaluation assistant. "
    "Given a question, ground truth items, and AI answer, extract identified items from the AI answer "
    "and score each ground-truth item in [0,1] by semantic match quality. "
    "Return JSON only."
)

UNSURE_ANSWERS = {
    "unsure",
    "not sure",
    "unknown",
    "cannot determine",
    "can't determine",
    "i don't know",
    "dont know",
    "i am not sure",
    "i'm not sure",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge answers with OLMo3 GGUF and generate metrics/plots.")
    p.add_argument("--answers-jsonl", action="append", default=[], help="Input answers JSONL (repeat).")
    p.add_argument("--judged-jsonl", action="append", default=[], help="Existing judged JSONL (repeat).")
    p.add_argument("--qa-csv", type=Path, default=None, help="Optional QA CSV fallback.")
    p.add_argument("--output-dir", type=Path, default=Path("refactoring_files") / "artifacts" / "judge_olmo")

    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--api-key", default="not-needed")
    p.add_argument("--judge-model", default="Olmo-3-7B-Instruct")
    p.add_argument("--max-judge-tokens", type=int, default=512)
    p.add_argument("--timeout-sec", type=int, default=120)
    p.add_argument("--request-retries", type=int, default=3)
    p.add_argument("--request-backoff-sec", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5, help="Match threshold for TP from matching scores.")
    p.add_argument(
        "--fast-unsure-shortcut",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, score unsure/unknown answers with non-empty gold as deterministic zero-match (faster).",
    )

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--report-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def parse_structured_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, dict):
        vals = list(value.values())
        if vals:
            return [str(x).strip() for x in vals if str(x).strip()]
        return [str(k).strip() for k in value.keys() if str(k).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.lower() in {"set()", "[]", "{}", "none", "null"}:
            return []
        # Try JSON then python-literal.
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(s)
                return parse_structured_items(parsed)
            except Exception:
                pass
        # Fallback: split comma lists if obvious, else single item.
        if "," in s:
            parts = [x.strip() for x in s.split(",")]
            clean = [p for p in parts if p]
            if clean:
                return clean
        return [s]
    return [str(value).strip()]


def parse_json_from_text(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    else:
        blob = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if blob:
            t = blob.group(0).strip()
    try:
        return json.loads(t)
    except Exception:
        return {}


def call_judge_model(
    client: OpenAI,
    model: str,
    question: str,
    retrieval_type: Optional[str],
    get_mode: Optional[str],
    ground_truth_items: List[str],
    answer: str,
    max_tokens: int,
    retries: int,
    backoff_sec: float,
) -> str:
    user_prompt = (
        "Evaluate the AI answer against the ground-truth items.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieval type: {retrieval_type or 'n/a'}\n"
        f"Get mode: {get_mode or 'n/a'}\n\n"
        f"Ground-truth items (order matters for score list):\n{json.dumps(ground_truth_items, ensure_ascii=False)}\n\n"
        f"AI answer:\n{answer}\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "identified_items_in_ai_answer": [string, ...],\n'
        '  "matching_scores": [float, ...],\n'
        '  "explanation": "short explanation"\n'
        "}\n"
        "Rules:\n"
        "- matching_scores length must equal number of ground-truth items.\n"
        "- Use scores in [0,1]. 1=exact, 0=no match.\n"
        "- If no ground-truth items, return empty matching_scores.\n"
        "- Return JSON only."
    )

    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            out = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return (out.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(backoff_sec * (2 ** (attempt - 1)))
    raise last_exc if last_exc else RuntimeError("Unknown judge call failure")


def normalize_judge_result(raw_text: str, n_gold: int) -> Tuple[List[str], List[float], str]:
    parsed = parse_json_from_text(raw_text)
    identified = parse_structured_items(parsed.get("identified_items_in_ai_answer", []))

    scores_raw = parsed.get("matching_scores", [])
    scores: List[float] = []
    if isinstance(scores_raw, list):
        for s in scores_raw:
            try:
                x = float(s)
            except Exception:
                x = 0.0
            if math.isnan(x) or math.isinf(x):
                x = 0.0
            scores.append(max(0.0, min(1.0, x)))

    # Force alignment to gold length.
    if len(scores) < n_gold:
        scores.extend([0.0] * (n_gold - len(scores)))
    if len(scores) > n_gold:
        scores = scores[:n_gold]

    explanation = str(parsed.get("explanation", "")).strip()
    return identified, scores, explanation


def calc_prf(identified: List[str], scores: List[float], threshold: float) -> Tuple[float, float, float, int]:
    n_pred = len(identified)
    n_gold = len(scores)
    tp = sum(1 for s in scores if s >= threshold)

    if n_gold == 0:
        precision = 1.0 if n_pred == 0 else 0.0
        recall = 1.0
        f1 = 1.0 if n_pred == 0 else 0.0
        return precision, recall, f1, tp

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gold
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, tp


def row_key_from_any(row: Dict[str, Any]) -> str:
    rk = row.get("row_key")
    if rk:
        return str(rk)
    if row.get("combined_q_idx") is not None:
        return f"combined_q_idx:{int(row['combined_q_idx'])}"
    if row.get("q_idx") is not None:
        return f"q_idx:{int(row['q_idx'])}"
    if row.get("row_idx") is not None:
        return f"row_idx:{int(row['row_idx'])}"
    return f"row_idx:0"


def load_qa_index(qa_csv: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(qa_csv)
    key_to_idx: Dict[str, int] = {}
    for i, r in df.iterrows():
        if "combined_q_idx" in df.columns and pd.notna(r.get("combined_q_idx")):
            key_to_idx[f"combined_q_idx:{int(r['combined_q_idx'])}"] = i
        if "q_idx" in df.columns and pd.notna(r.get("q_idx")):
            key_to_idx.setdefault(f"q_idx:{int(r['q_idx'])}", i)
        key_to_idx.setdefault(f"row_idx:{int(i)}", i)
    return df, key_to_idx


def enrich_from_qa(row: Dict[str, Any], qa_df: Optional[pd.DataFrame], qa_idx: Optional[Dict[str, int]]) -> Dict[str, Any]:
    if qa_df is None or qa_idx is None:
        return row
    rk = row_key_from_any(row)
    if rk not in qa_idx:
        return row
    qr = qa_df.iloc[qa_idx[rk]]
    for col in ("minerva_category", "missing_tuple_elements", "cue", "retrieval_type", "get", "correct_answer", "question"):
        if col in qa_df.columns and (row.get(col) is None or row.get(col) == ""):
            v = qr.get(col)
            if pd.notna(v):
                row[col] = v
    return row


def infer_qa_csv(answer_rows: List[Dict[str, Any]], explicit_qa_csv: Optional[Path]) -> Optional[Path]:
    if explicit_qa_csv is not None:
        return explicit_qa_csv
    for r in answer_rows:
        q = r.get("qa_csv")
        if q and Path(str(q)).exists():
            return Path(str(q))
    return None


def judge_one_answers_file(
    answers_jsonl: Path,
    qa_csv: Optional[Path],
    output_dir: Path,
    client: OpenAI,
    args: argparse.Namespace,
) -> Path:
    answers = read_jsonl(answers_jsonl)
    if args.max_rows is not None:
        answers = answers[: args.max_rows]

    if not answers:
        raise ValueError(f"No rows in answers file: {answers_jsonl}")

    qa_path = infer_qa_csv(answers, qa_csv)
    qa_df: Optional[pd.DataFrame] = None
    qa_index: Optional[Dict[str, int]] = None
    if qa_path is not None and qa_path.exists():
        qa_df, qa_index = load_qa_index(qa_path)

    judged_path = output_dir / f"{answers_jsonl.stem}.judged.olmo.jsonl"
    done_keys = set()
    if args.resume and judged_path.exists():
        for r in read_jsonl(judged_path):
            if r.get("row_key"):
                done_keys.add(str(r["row_key"]))

    for i, row in enumerate(answers):
        rk = row_key_from_any(row)
        if rk in done_keys:
            continue

        row = enrich_from_qa(dict(row), qa_df, qa_index)
        question = str(row.get("question", ""))
        retrieval_type = row.get("retrieval_type")
        get_mode = row.get("get")
        # Prefer explicit answer field, but if it is blank, fall back to llm_answer.
        answer_raw = row.get("answer")
        if answer_raw is None or str(answer_raw).strip() == "":
            answer_raw = row.get("llm_answer", "")
        answer = str(answer_raw or "")

        gt_items = parse_structured_items(row.get("correct_answer"))
        # fallback: some files may store ground truth in dedicated field.
        if not gt_items and row.get("ground_truth_items") is not None:
            gt_items = parse_structured_items(row.get("ground_truth_items"))

        judge_status = "ok"
        judge_error: Optional[str] = None
        raw_judge_text = ""
        identified: List[str] = []
        scores: List[float] = [0.0] * len(gt_items)
        explanation = ""

        answer_norm = answer.strip().lower()

        # Deterministic safeguard: empty answers must never receive positive matches.
        if answer.strip() == "":
            identified = []
            scores = [0.0] * len(gt_items)
            if len(gt_items) == 0:
                explanation = "Empty answer with empty ground truth; deterministic exact match."
            else:
                explanation = "Empty answer with non-empty ground truth; deterministic zero match."
            judge_status = "ok"
        # Fast path for fallback answers: if model says "unsure" and gold is non-empty,
        # this cannot be a true positive match, so skip expensive judge calls.
        elif args.fast_unsure_shortcut and answer_norm in UNSURE_ANSWERS and len(gt_items) > 0:
            identified = []
            scores = [0.0] * len(gt_items)
            explanation = "Unsure/unknown answer with non-empty ground truth; deterministic zero match."
            judge_status = "ok"
        else:
            try:
                raw_judge_text = call_judge_model(
                    client=client,
                    model=args.judge_model,
                    question=question,
                    retrieval_type=str(retrieval_type) if retrieval_type is not None else None,
                    get_mode=str(get_mode) if get_mode is not None else None,
                    ground_truth_items=gt_items,
                    answer=answer,
                    max_tokens=args.max_judge_tokens,
                    retries=args.request_retries,
                    backoff_sec=args.request_backoff_sec,
                )
                identified, scores, explanation = normalize_judge_result(raw_judge_text, len(gt_items))
            except Exception as exc:  # noqa: BLE001
                judge_status = "error"
                judge_error = f"{type(exc).__name__}: {exc}"
                explanation = judge_error

        precision, recall, f1, tp = calc_prf(identified, scores, args.threshold)
        exact_match = 1 if ((len(gt_items) == 0 and len(identified) == 0) or (len(gt_items) > 0 and tp == len(gt_items) and len(identified) == len(gt_items))) else 0

        out = {
            "source_answers_file": str(answers_jsonl),
            "timestamp_utc": utc_now(),
            "mode": row.get("mode"),
            "row_key": rk,
            "row_idx": row.get("row_idx", i),
            "combined_q_idx": row.get("combined_q_idx"),
            "q_idx": row.get("q_idx"),
            "question": question,
            "retrieval_type": retrieval_type,
            "get": get_mode,
            "ground_truth_items": gt_items,
            "answer": answer,
            "identified_items": identified,
            "matching_scores": scores,
            "tp": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "exact_match": int(exact_match),
            "judge_explanation": explanation,
            "judge_model": args.judge_model,
            "judge_status": judge_status,
            "status": judge_status,  # compatibility alias
            "judge_error": judge_error,
            "raw_judge_text": raw_judge_text,
            "minerva_category": row.get("minerva_category"),
            "missing_tuple_elements": row.get("missing_tuple_elements"),
            "source_status": row.get("status"),
        }
        append_jsonl(judged_path, out)
        done_keys.add(rk)

    return judged_path


def cardinality_bin(n_items: int) -> str:
    if n_items <= 0:
        return "0"
    if n_items == 1:
        return "1"
    if n_items == 2:
        return "2"
    if n_items <= 5:
        return "3-5"
    return "6+"


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def make_plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, top_n: Optional[int] = None) -> None:
    if plt is None:
        return
    d = df.copy()
    d = d.sort_values(y, ascending=False)
    if top_n is not None:
        d = d.head(top_n)
    plt.figure(figsize=(10, 5))
    plt.bar(d[x].astype(str), d[y].astype(float))
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_plot_hist(df: pd.DataFrame, col: str, title: str, out_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(df[col].astype(float), bins=20)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def summarize_and_plot(judged_paths: Sequence[Path], output_dir: Path, make_plots: bool) -> None:
    all_rows: List[Dict[str, Any]] = []
    for p in judged_paths:
        all_rows.extend(read_jsonl(p))
    if not all_rows:
        raise ValueError("No judged rows found to summarize.")

    df = pd.DataFrame(all_rows)
    for col in ("f1", "precision", "recall"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    if "exact_match" not in df.columns:
        df["exact_match"] = (df["f1"] >= 0.9999).astype(int)
    else:
        df["exact_match"] = pd.to_numeric(df["exact_match"], errors="coerce").fillna(0).astype(int)

    df["n_gold"] = df["ground_truth_items"].apply(lambda x: len(parse_structured_items(x)))
    df["gold_bin"] = df["n_gold"].apply(cardinality_bin)
    df["source_file"] = df["source_answers_file"].apply(lambda s: Path(str(s)).name if pd.notna(s) else "unknown")
    df["retrieval_type"] = df["retrieval_type"].fillna("unknown")
    df["minerva_category"] = df["minerva_category"].fillna("unknown")
    df["get"] = df["get"].fillna("").astype(str)

    # Overall metrics.
    overall = pd.DataFrame(
        [
            {
                "n_rows": int(len(df)),
                "mean_f1": round(float(df["f1"].mean()), 6),
                "median_f1": round(float(df["f1"].median()), 6),
                "mean_precision": round(float(df["precision"].mean()), 6),
                "mean_recall": round(float(df["recall"].mean()), 6),
                "exact_match_rate": round(float(df["exact_match"].mean()), 6),
            }
        ]
    )
    overall.to_csv(output_dir / "overall_metrics.csv", index=False)

    by_file = (
        df.groupby("source_file", as_index=False)
        .agg(n_rows=("f1", "size"), mean_f1=("f1", "mean"), exact_match_rate=("exact_match", "mean"))
        .sort_values("mean_f1", ascending=False)
    )
    by_file.to_csv(output_dir / "metrics_by_file.csv", index=False)

    by_retrieval = (
        df.groupby("retrieval_type", as_index=False)
        .agg(n_rows=("f1", "size"), mean_f1=("f1", "mean"), exact_match_rate=("exact_match", "mean"))
        .sort_values("mean_f1", ascending=False)
    )
    by_retrieval.to_csv(output_dir / "metrics_by_retrieval_type.csv", index=False)

    by_minerva = (
        df.groupby("minerva_category", as_index=False)
        .agg(n_rows=("f1", "size"), mean_f1=("f1", "mean"), exact_match_rate=("exact_match", "mean"))
        .sort_values("mean_f1", ascending=False)
    )
    by_minerva.to_csv(output_dir / "metrics_by_minerva_category.csv", index=False)

    by_bin = (
        df.groupby("gold_bin", as_index=False)
        .agg(n_rows=("f1", "size"), mean_f1=("f1", "mean"))
        .sort_values("gold_bin")
    )
    by_bin.to_csv(output_dir / "metrics_by_gold_cardinality_bin.csv", index=False)

    # Benchmark-like high-level scores.
    simple_recall_score = safe_mean(by_bin["mean_f1"]) if not by_bin.empty else float("nan")
    latest_mask = df["get"].str.lower().eq("latest") | df["retrieval_type"].str.lower().str.contains("latest", na=False)
    chrono_mask = df["get"].str.lower().str.contains("chron", na=False) | df["retrieval_type"].str.lower().str.contains("chron", na=False)

    latest_state_score = float(df.loc[latest_mask, "f1"].mean()) if latest_mask.any() else float("nan")
    chronological_order_score = float(df.loc[chrono_mask, "f1"].mean()) if chrono_mask.any() else float("nan")

    ca_parts = [x for x in (latest_state_score, chronological_order_score) if not math.isnan(x)]
    chronological_awareness_score = float(sum(ca_parts) / len(ca_parts)) if ca_parts else float("nan")

    score_summary = pd.DataFrame(
        [
            {
                "simple_recall_score": round(simple_recall_score, 6) if not math.isnan(simple_recall_score) else "",
                "latest_state_score": round(latest_state_score, 6) if not math.isnan(latest_state_score) else "",
                "chronological_order_score": round(chronological_order_score, 6)
                if not math.isnan(chronological_order_score)
                else "",
                "chronological_awareness_score": round(chronological_awareness_score, 6)
                if not math.isnan(chronological_awareness_score)
                else "",
            }
        ]
    )
    score_summary.to_csv(output_dir / "score_summary.csv", index=False)

    # Save merged judged table for easy downstream analysis.
    df.to_csv(output_dir / "judged_merged_rows.csv", index=False)

    if make_plots and plt is not None:
        make_plot_hist(df, "f1", "F1 Distribution", output_dir / "plot_f1_hist.png")
        make_plot_bar(by_file, "source_file", "mean_f1", "Mean F1 by Source File", output_dir / "plot_mean_f1_by_file.png")
        make_plot_bar(
            by_retrieval,
            "retrieval_type",
            "mean_f1",
            "Mean F1 by Retrieval Type",
            output_dir / "plot_mean_f1_by_retrieval_type.png",
            top_n=20,
        )
        make_plot_bar(
            by_minerva,
            "minerva_category",
            "mean_f1",
            "Mean F1 by Minerva Category",
            output_dir / "plot_mean_f1_by_minerva_category.png",
            top_n=20,
        )
        make_plot_bar(
            by_file,
            "source_file",
            "exact_match_rate",
            "Exact-Match Rate by Source File",
            output_dir / "plot_em_by_file.png",
        )


def ensure_openai_available() -> None:
    if OpenAI is None:
        raise ModuleNotFoundError("openai package is required. Install with: python -m pip install openai")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    judged_paths: List[Path] = [Path(p).resolve() for p in args.judged_jsonl]

    if not args.report_only:
        ensure_openai_available()
        answers_paths = [Path(p).resolve() for p in args.answers_jsonl]
        if not answers_paths:
            raise ValueError("Provide at least one --answers-jsonl when not using --report-only.")

        qa_csv = args.qa_csv.resolve() if args.qa_csv is not None else None
        client = OpenAI(
            base_url=normalize_base_url(args.base_url),
            api_key=args.api_key,
            timeout=args.timeout_sec,
        )
        for p in answers_paths:
            judged_paths.append(
                judge_one_answers_file(
                    answers_jsonl=p,
                    qa_csv=qa_csv,
                    output_dir=output_dir,
                    client=client,
                    args=args,
                )
            )

    if not judged_paths:
        raise ValueError("No judged files available for summary. Provide --judged-jsonl or run judging first.")

    summarize_and_plot(judged_paths=judged_paths, output_dir=output_dir, make_plots=(not args.no_plots))
    print(f"[{utc_now()}] done. outputs in: {output_dir}")


if __name__ == "__main__":
    main()
