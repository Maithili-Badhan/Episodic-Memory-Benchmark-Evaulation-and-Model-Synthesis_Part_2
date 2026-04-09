#!/usr/bin/env python
"""
DP-only runner for Colab T4 using 4-bit quantized Transformers.

Why this script:
- Avoids vLLM dependency/CUDA wheel friction in notebook runtimes.
- Fits Qwen2.5-14B on a 16 GB T4 using bitsandbytes 4-bit quantization.
- Optional LoRA adapter loading (PEFT).

Default input layout is /content with flat filenames:
- /content/news_book.txt
- /content/scifi_book.txt
- /content/book.txt   (thriller)
- /content/news_news_llm_retry_df_qa.combined_story_minerva_balanced_150.csv
- /content/scifi_20260312T170655Z_df_qa.combined_story_minerva_balanced_150.csv
- /content/thriller_20260311T152054Z_df_qa.combined_story_minerva_balanced_150.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except ModuleNotFoundError:
    PeftModel = None


SYSTEM_PROMPT = "You are an expert in memory tests."
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def utc_tag() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
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


def default_jobs(base_dir: Path, preset: str) -> List[Dict[str, Path | str]]:
    if preset not in {"minerva150", "minimal150"}:
        raise ValueError(f"Unsupported preset: {preset}")
    qa_suffix = "minerva_balanced" if preset == "minerva150" else "minimal"
    return [
        {
            "name": "news",
            "book_txt": base_dir / "news_book.txt",
            "qa_csv": base_dir / f"news_news_llm_retry_df_qa.combined_story_{qa_suffix}_150.csv",
        },
        {
            "name": "scifi",
            "book_txt": base_dir / "scifi_book.txt",
            "qa_csv": base_dir / f"scifi_20260312T170655Z_df_qa.combined_story_{qa_suffix}_150.csv",
        },
        {
            "name": "thriller",
            "book_txt": base_dir / "book.txt",
            "qa_csv": base_dir / f"thriller_20260311T152054Z_df_qa.combined_story_{qa_suffix}_150.csv",
        },
    ]


def parse_job_arg(job_arg: str) -> Dict[str, Path | str]:
    parts = [p.strip() for p in job_arg.split("|")]
    if len(parts) != 3:
        raise ValueError(f"Invalid --job format: {job_arg}")
    return {"name": parts[0], "book_txt": Path(parts[1]), "qa_csv": Path(parts[2])}


def build_dp_prompt(question: str, book_text: str, max_chars: Optional[int]) -> str:
    src = book_text
    if max_chars is not None and max_chars > 0:
        src = src[:max_chars]
    return f"""# Episodic Memory Benchmark
You are participating in an episodic memory test.
Answer using only the book text. If unsure, say unsure.

## Book Text:
{src}

## Question:
{question}

Return only the final answer."""


def render_chat(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_tokens = int(enc["input_ids"].shape[1])

    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][prompt_tokens:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, prompt_tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-14B DP-only 4-bit runner for Colab T4.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--lora-path", type=str, default="", help="Optional HF Hub id or local path to LoRA adapter.")
    p.add_argument("--preset", choices=["minerva150", "minimal150"], default="minerva150")
    p.add_argument("--base-dir", type=Path, default=Path("/content"), help="Base folder for default flat input files.")
    p.add_argument("--job", action="append", default=[], help="Optional custom job: name|book_txt|qa_csv")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content") / "outputs_qwen25_14b_dp_4bit",
    )
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dp-max-book-chars", type=int, default=12000)
    p.add_argument("--max-input-tokens", type=int, default=3072)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    # In notebooks, ipykernel injects args like: -f <kernel.json>.
    # parse_known_args keeps CLI usage intact while ignoring notebook extras.
    args, _unknown = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    # Notebook execution (copy-paste) may not define __file__.
    if "__file__" in globals():
        repo_root = Path(__file__).resolve().parents[2]
    else:
        repo_root = Path.cwd()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [parse_job_arg(j) for j in args.job] if args.job else default_jobs(args.base_dir.resolve(), args.preset)
    for job in jobs:
        job["book_txt"] = (repo_root / job["book_txt"]).resolve() if isinstance(job["book_txt"], Path) and not job["book_txt"].is_absolute() else Path(job["book_txt"]).resolve()
        job["qa_csv"] = (repo_root / job["qa_csv"]).resolve() if isinstance(job["qa_csv"], Path) and not job["qa_csv"].is_absolute() else Path(job["qa_csv"]).resolve()

    print(f"[{utc_now()}] CUDA available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required (Colab T4 expected).")
    print(f"[{utc_now()}] GPU={torch.cuda.get_device_name(0)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.lora_path:
        if PeftModel is None:
            raise ModuleNotFoundError("peft is required for --lora-path. Install: pip install peft")
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"[{utc_now()}] Loaded LoRA adapter: {args.lora_path}")

    manifest = {
        "timestamp_utc": utc_now(),
        "model_id": args.model_id,
        "lora_path": args.lora_path or None,
        "mode": "dp",
        "output_dir": str(output_dir),
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "dp_max_book_chars": args.dp_max_book_chars,
            "max_input_tokens": args.max_input_tokens,
            "max_rows": args.max_rows,
            "resume": args.resume,
        },
        "jobs": [{"name": j["name"], "book_txt": str(j["book_txt"]), "qa_csv": str(j["qa_csv"])} for j in jobs],
    }
    manifest_path = output_dir / f"qwen25_14b_t4_dp_manifest_{utc_tag()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{utc_now()}] manifest={manifest_path}")

    for job in jobs:
        name = str(job["name"])
        book_txt = Path(job["book_txt"])
        qa_csv = Path(job["qa_csv"])
        out_jsonl = output_dir / f"{name}_qwen25_14b_dp.answers.jsonl"

        if not book_txt.exists():
            raise FileNotFoundError(f"Missing book file: {book_txt}")
        if not qa_csv.exists():
            raise FileNotFoundError(f"Missing QA csv: {qa_csv}")

        book_text = book_txt.read_text(encoding="utf-8", errors="replace")
        df = pd.read_csv(qa_csv)
        if args.max_rows is not None:
            df = df.head(args.max_rows).copy()

        done = set()
        if args.resume and out_jsonl.exists():
            for row in read_jsonl(out_jsonl):
                if row.get("status") == "ok" and row.get("row_key"):
                    done.add(row["row_key"])

        print(f"[{utc_now()}] job={name} rows_total={len(df)} resume_done={len(done)}")
        for i in tqdm(range(len(df)), desc=f"{name}:dp"):
            row = df.iloc[i]
            row_key = f"combined_q_idx:{int(row['combined_q_idx'])}" if "combined_q_idx" in row and pd.notna(row["combined_q_idx"]) else f"row_idx:{i}"
            if row_key in done:
                continue

            question = str(row["question"])
            prompt = build_dp_prompt(question, book_text, args.dp_max_book_chars)
            rendered = render_chat(tokenizer, prompt)

            status = "ok"
            err = None
            answer = ""
            prompt_tokens = None
            t0 = time.time()
            try:
                answer, prompt_tokens = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rendered,
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                status = "error"
                err = f"{type(exc).__name__}: {exc}"

            rec = {
                "timestamp_utc": utc_now(),
                "status": status,
                "error": err,
                "mode": "dp",
                "model_id": args.model_id,
                "lora_path": args.lora_path or None,
                "row_key": row_key,
                "row_idx": int(i),
                "q_idx": int(row["q_idx"]) if "q_idx" in row and pd.notna(row["q_idx"]) else int(i),
                "combined_q_idx": int(row["combined_q_idx"]) if "combined_q_idx" in row and pd.notna(row["combined_q_idx"]) else None,
                "question": question,
                "retrieval_type": str(row["retrieval_type"]) if "retrieval_type" in row and pd.notna(row["retrieval_type"]) else None,
                "get": str(row["get"]) if "get" in row and pd.notna(row["get"]) else None,
                "correct_answer": str(row["correct_answer"]) if "correct_answer" in row and pd.notna(row["correct_answer"]) else None,
                "prompt_tokens": prompt_tokens,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "latency_s": round(time.time() - t0, 3),
                "answer": answer,
            }
            append_jsonl(out_jsonl, rec)
            if status == "ok":
                done.add(row_key)

        print(f"[{utc_now()}] done job={name} written={len(done)} output={out_jsonl}")

    print(f"[{utc_now()}] all jobs done")


if __name__ == "__main__":
    main()
