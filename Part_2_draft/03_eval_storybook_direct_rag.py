#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run direct prompting on storybook QA with a llama.cpp OpenAI-compatible server."
    )
    p.add_argument(
        "--storybook-dirs",
        nargs="+",
        required=True,
        help="One or more storybook artifact directories (containing book.txt and df_qa.*).",
    )
    p.add_argument("--llamacpp-base-url", type=str, default="http://127.0.0.1:8080/")
    p.add_argument("--api-key", type=str, default="not-needed")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--mode", choices=["direct"], default="direct")
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("refactoring_files") / "artifacts" / "eval_direct",
    )
    p.add_argument("--run-name", type=str, default="")
    return p.parse_args()


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def generate_episodic_memory_prompt(book_content: str, question: str) -> str:
    return f"""# Episodic Memory Benchmark

You are participating in an episodic memory test. You will be presented with a text to read and internalize as if you had personally experienced the events described. After the text, you will find a question about the content. Please answer this question based solely on the information provided in the text.

## The Text to Memorize:

{book_content}

## Question:

{question}

Please answer the question to the best of your ability, based only on the information provided in the text above. If you are unsure about an answer, it's okay to say so. Do not invent or assume information that was not explicitly stated in the text.
"""


def load_df_qa(storybook_dir: Path) -> pd.DataFrame:
    parquet_path = storybook_dir / "df_qa.parquet"
    jsonl_path = storybook_dir / "df_qa.jsonl"
    csv_path = storybook_dir / "df_qa.csv"

    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)
    if jsonl_path.is_file():
        return pd.read_json(jsonl_path, lines=True)
    if csv_path.is_file():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"No QA file found under {storybook_dir}. Expected one of: "
        "df_qa.parquet, df_qa.jsonl, df_qa.csv"
    )


def normalize_assistant_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def parse_structured(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    if s[0] not in "[{(":
        return value
    try:
        return ast.literal_eval(s)
    except Exception:
        return value


def load_existing_qidx(output_jsonl: Path) -> set[int]:
    if not output_jsonl.is_file():
        return set()
    done: set[int] = set()
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q_idx = row.get("q_idx")
            if isinstance(q_idx, int):
                done.add(q_idx)
    return done


def answer_one_question(
    client: OpenAI,
    model_name: str,
    book_text: str,
    question: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int | None, int | None]:
    user_prompt = generate_episodic_memory_prompt(book_text, question)
    out = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert in memory tests."},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    answer = normalize_assistant_content(out.choices[0].message.content)
    usage = getattr(out, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    return answer, prompt_tokens, completion_tokens


def main() -> None:
    args = parse_args()
    base_url = normalize_base_url(args.llamacpp_base_url)
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_dir.resolve() / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "mode": args.mode,
        "llamacpp_base_url": base_url,
        "model_name": args.model_name,
        "max_questions": args.max_questions,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "sleep_seconds": args.sleep_seconds,
        "storybook_dirs": [str(Path(x).resolve()) for x in args.storybook_dirs],
        "results": [],
    }

    for storybook_dir_raw in args.storybook_dirs:
        storybook_dir = Path(storybook_dir_raw).resolve()
        if not storybook_dir.is_dir():
            raise FileNotFoundError(f"Storybook dir not found: {storybook_dir}")

        book_path = storybook_dir / "book.txt"
        if not book_path.is_file():
            raise FileNotFoundError(f"Missing book.txt in {storybook_dir}")
        book_text = book_path.read_text(encoding="utf-8")

        df_qa = load_df_qa(storybook_dir)
        if "question" not in df_qa.columns:
            raise ValueError(f"df_qa in {storybook_dir} does not contain 'question' column")

        if args.max_questions is not None:
            df_qa = df_qa.head(args.max_questions)

        safe_name = storybook_dir.name
        out_jsonl = output_root / f"{safe_name}.answers.jsonl"
        out_csv = output_root / f"{safe_name}.answers.csv"

        if args.overwrite:
            if out_jsonl.exists():
                out_jsonl.unlink()
            if out_csv.exists():
                out_csv.unlink()

        done_q = load_existing_qidx(out_jsonl) if args.resume and not args.overwrite else set()
        rows: list[dict[str, Any]] = []
        if out_jsonl.is_file() and args.resume and not args.overwrite:
            with out_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        total = len(df_qa)
        print(f"[{safe_name}] total questions to consider: {total}, already done: {len(done_q)}")
        with out_jsonl.open("a", encoding="utf-8") as f:
            for i, qa_row in df_qa.reset_index(drop=True).iterrows():
                q_idx = int(qa_row.get("q_idx", i))
                if q_idx in done_q:
                    continue

                question = str(qa_row["question"])
                started = time.perf_counter()
                answer, prompt_tokens, completion_tokens = answer_one_question(
                    client=client,
                    model_name=args.model_name,
                    book_text=book_text,
                    question=question,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                elapsed = time.perf_counter() - started

                out_row: dict[str, Any] = {
                    "storybook_dir": str(storybook_dir),
                    "q_idx": q_idx,
                    "question_row_index": int(i),
                    "question": question,
                    "retrieval_type": to_jsonable(qa_row.get("retrieval_type")),
                    "get": to_jsonable(qa_row.get("get")),
                    "correct_answer": to_jsonable(parse_structured(qa_row.get("correct_answer"))),
                    "llm_answer": answer,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_s": round(elapsed, 4),
                    "model_name": args.model_name,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }

                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                rows.append(out_row)
                print(f"[{safe_name}] q_idx={q_idx} done in {elapsed:.2f}s")

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out.to_csv(out_csv, index=False)

        manifest["results"].append(
            {
                "storybook_dir": str(storybook_dir),
                "output_jsonl": str(out_jsonl.resolve()),
                "output_csv": str(out_csv.resolve()),
                "rows_written": int(len(rows)),
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Run outputs: {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
