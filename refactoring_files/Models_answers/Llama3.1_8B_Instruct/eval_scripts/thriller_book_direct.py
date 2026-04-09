#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate storybook QA with direct prompting and save Qwen-like JSONL/CSV outputs."
    )
    p.add_argument(
        "--storybook-dirs",
        nargs="+",
        default=None,
        help="One or more storybook dirs (or parent dirs) containing book.txt and df_qa.*",
    )
    p.add_argument("--qa-file", type=str, default=None, help="Single QA file (csv/jsonl/parquet)")
    p.add_argument("--book-file", type=str, default=None, help="Single book text file")
    p.add_argument("--llamacpp-base-url", type=str, default="http://127.0.0.1:8081")
    p.add_argument("--api-key", type=str, default="not-needed")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--prompt-style", type=str, default="detailed", choices=["brief", "detailed"])
    p.add_argument("--max-chars", type=int, default=12000)

    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max-questions", type=int, default=None)

    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep-seconds", type=float, default=0.0)

    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--out-root", type=str, default="artifacts/storybooks")
    return p.parse_args()


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def get_first_available(row: pd.Series, keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def load_df_qa(storybook_dir: Path) -> tuple[pd.DataFrame, Path]:
    for name, loader in (
        ("df_qa.parquet", pd.read_parquet),
        ("df_qa.jsonl", lambda p: pd.read_json(p, lines=True)),
        ("df_qa.csv", pd.read_csv),
        ("qa_from_book.csv", pd.read_csv),
    ):
        fp = storybook_dir / name
        if fp.is_file():
            return loader(fp), fp
    raise FileNotFoundError(f"No QA file found under {storybook_dir}")


def load_df_qa_file(qa_file: Path) -> pd.DataFrame:
    suffix = qa_file.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(qa_file)
    if suffix in (".jsonl", ".json"):
        return pd.read_json(qa_file, lines=True)
    if suffix == ".csv":
        return pd.read_csv(qa_file)
    raise ValueError(f"Unsupported QA file format: {qa_file}")


def is_storybook_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "book.txt").is_file():
        return False
    return any(
        (path / name).is_file()
        for name in ("df_qa.parquet", "df_qa.jsonl", "df_qa.csv", "qa_from_book.csv")
    )


def resolve_storybook_dirs(raw_paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in raw_paths:
        p = Path(raw)
        if is_storybook_dir(p):
            found.append(p.resolve())
            continue
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if is_storybook_dir(child):
                    found.append(child.resolve())
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def generate_prompt(book_text: str, question: str, max_chars: int, prompt_style: str) -> str:
    context = book_text[:max_chars]
    if prompt_style == "brief":
        return (
            "Answer ONLY from the provided text."
            f"\n\nTEXT:\n{context}"
            f"\n\nQUESTION:\n{question}"
            "\n\nReturn only the final answer."
        )
    return (
        "You are evaluating episodic memory QA.\n"
        "Use ONLY the provided text and follow query instructions exactly.\n"
        "If the query asks for a set/list/intersection/difference/sorted output, return only that result.\n"
        "Do not add explanation.\n"
        f"\nTEXT:\n{context}\n"
        f"\nQUESTION:\n{question}\n"
        "\nFINAL ANSWER:"
    )


def ask_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    out = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a precise QA model."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (out.choices[0].message.content or "").strip()


def load_processed_keys(jsonl_path: Path) -> set[str]:
    keys: set[str] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("row_key")
                if key:
                    keys.add(str(key))
            except Exception:
                continue
    return keys


def main() -> None:
    args = parse_args()

    run_id = args.run_name or f"llama_direct_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=normalize_base_url(args.llamacpp_base_url), api_key=args.api_key)

    eval_items: list[tuple[str, Path, pd.DataFrame, Path]] = []
    if args.qa_file or args.book_file:
        if not (args.qa_file and args.book_file):
            raise SystemExit("Provide both --qa-file and --book-file for single-file mode.")
        qa_path = Path(args.qa_file).resolve()
        book_path = Path(args.book_file).resolve()
        if not qa_path.is_file():
            raise SystemExit(f"QA file not found: {qa_path}")
        if not book_path.is_file():
            raise SystemExit(f"Book file not found: {book_path}")
        df = load_df_qa_file(qa_path)
        eval_items.append((qa_path.stem, book_path, df, qa_path))
    else:
        story_dirs = resolve_storybook_dirs(args.storybook_dirs or [])
        if not story_dirs:
            raise SystemExit("No valid storybook directories found.")
        for story_dir in story_dirs:
            book_path = story_dir / "book.txt"
            df, qa_path = load_df_qa(story_dir)
            eval_items.append((story_dir.name, book_path, df, qa_path))

    for eval_name, book_path, df, qa_path in eval_items:
        book_text = book_path.read_text(encoding="utf-8")

        if "question" not in df.columns:
            raise SystemExit(f"Missing 'question' column in {qa_path}")

        if args.end is not None:
            df = df.iloc[args.start : args.end]
        else:
            df = df.iloc[args.start :]
        if args.max_questions is not None:
            df = df.iloc[: args.max_questions]
        df = df.reset_index(drop=True)

        out_jsonl = out_root / f"{eval_name}.{run_id}.answers.jsonl"
        out_csv = out_root / f"{eval_name}.{run_id}.answers.csv"
        checkpoint_csv = out_root / f"{eval_name}.{run_id}.checkpoint.csv"

        if args.overwrite:
            out_jsonl.unlink(missing_ok=True)
            out_csv.unlink(missing_ok=True)
            checkpoint_csv.unlink(missing_ok=True)
        elif out_jsonl.exists() and not args.resume:
            raise SystemExit(
                f"Output exists: {out_jsonl}. Use --resume to continue or --overwrite to replace."
            )

        processed_keys = load_processed_keys(out_jsonl) if args.resume else set()
        rows_csv: list[dict[str, Any]] = []

        print(
            f"Running direct eval | dataset={eval_name} | n={len(df)} | out={out_jsonl.name}"
        )

        with out_jsonl.open("a", encoding="utf-8") as fj:
            for row_idx, row in df.iterrows():
                question = str(row.get("question", "")).strip()
                q_idx = int(get_first_available(row, ["q_idx", "id"], row_idx))
                combined_q_idx = int(get_first_available(row, ["combined_q_idx"], row_idx))
                row_key = f"combined_q_idx:{combined_q_idx}"

                if args.resume and row_key in processed_keys:
                    continue

                t0 = time.time()
                prompt = generate_prompt(book_text, question, args.max_chars, args.prompt_style)

                status = "ok"
                error = None
                answer = ""
                try:
                    answer = ask_model(
                        client=client,
                        model_name=args.model_name,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"

                latency_sec = round(time.time() - t0, 3)

                rec = {
                    "run_id": run_id,
                    "timestamp_utc": utc_now_iso(),
                    "status": status,
                    "error": error,
                    "qa_csv": str(qa_path.resolve()),
                    "book_txt": str(book_path.resolve()),
                    "row_key": row_key,
                    "row_idx": int(row_idx),
                    "q_idx": q_idx,
                    "combined_q_idx": combined_q_idx,
                    "question": question,
                    "cue": get_first_available(row, ["cue"]),
                    "retrieval_type": get_first_available(row, ["retrieval_type"]),
                    "get": get_first_available(row, ["get"]),
                    "correct_answer": to_jsonable(
                        get_first_available(row, ["correct_answer", "answer"])
                    ),
                    "llm_model": args.model_name,
                    "llm_base_url": normalize_base_url(args.llamacpp_base_url),
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_tokens,
                    "rag_embedding_model": None,
                    "rag_top_k": None,
                    "prompt_style": args.prompt_style,
                    "max_chars_per_chunk": args.max_chars,
                    "retrieved_chunks": [],
                    "answer": answer,
                    "latency_sec": latency_sec,
                }

                fj.write(json.dumps(rec, ensure_ascii=False) + "\n")

                rows_csv.append(
                    {
                        "row_key": row_key,
                        "q_idx": q_idx,
                        "combined_q_idx": combined_q_idx,
                        "question": question,
                        "correct_answer": to_jsonable(
                            get_first_available(row, ["correct_answer", "answer"])
                        ),
                        "answer": answer,
                        "status": status,
                        "error": error,
                        "latency_sec": latency_sec,
                    }
                )

                if args.checkpoint_every > 0 and (len(rows_csv) % args.checkpoint_every == 0):
                    pd.DataFrame(rows_csv).to_csv(checkpoint_csv, index=False)

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

        if out_jsonl.exists():
            json_rows = [json.loads(x) for x in out_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
            pd.DataFrame(
                {
                    "row_key": r.get("row_key"),
                    "q_idx": r.get("q_idx"),
                    "combined_q_idx": r.get("combined_q_idx"),
                    "question": r.get("question"),
                    "correct_answer": r.get("correct_answer"),
                    "answer": r.get("answer"),
                    "status": r.get("status"),
                    "error": r.get("error"),
                    "latency_sec": r.get("latency_sec"),
                }
                for r in json_rows
            ).to_csv(out_csv, index=False)

        print(f"Saved: {out_jsonl}")
        print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
