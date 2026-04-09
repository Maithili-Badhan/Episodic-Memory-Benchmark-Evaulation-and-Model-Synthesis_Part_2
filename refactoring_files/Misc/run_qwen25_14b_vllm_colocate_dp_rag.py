#!/usr/bin/env python
"""
Run Qwen2.5-14B direct prompting (DP) and RAG with vLLM in colocated mode.

Colocated mode here means we run inference through `vllm.LLM(...)` in-process
instead of a separate HTTP server, removing network/process overhead.

Example:
  python refactoring_files/qna_three_books/run_qwen25_14b_vllm_colocate_dp_rag.py \
    --preset minerva150 \
    --modes rag dp \
    --output-dir refactoring_files/Models_answers/Qwen2.5_14B_instruct \
    --max-new-tokens 80 \
    --temperature 0.0 \
    --batch-size 8
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import faiss
except Exception:  # noqa: BLE001
    # Covers both missing module and NumPy ABI mismatch import failures.
    faiss = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    AutoTokenizer = None

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    LLM = None
    SamplingParams = None


SYSTEM_PROMPT = "You are an expert in memory tests."
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def utc_run_tag() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log_line(message: str, log_path: Path) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def split_book_into_chunks(book_text: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    chapters = book_text.split("Chapter ")
    if len(chapters) > 1:
        for chapter_idx, chapter in enumerate(chapters[1:], start=1):
            paragraphs = [p.strip() for p in chapter.strip().split("\n\n") if p.strip()]
            for para_idx, para in enumerate(paragraphs, start=1):
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "label": f"Chapter {chapter_idx}, Paragraph {para_idx}",
                        "chapter": chapter_idx,
                        "paragraph": para_idx,
                        "text": para,
                    }
                )
    else:
        paragraphs = [p.strip() for p in book_text.split("\n\n") if p.strip()]
        for para_idx, para in enumerate(paragraphs, start=1):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "label": f"Paragraph {para_idx}",
                    "chapter": None,
                    "paragraph": para_idx,
                    "text": para,
                }
            )
    return chunks


def build_rag_index(book_text: str, embedder: SentenceTransformer) -> Tuple[Any, List[Dict[str, Any]], str]:
    chunks = split_book_into_chunks(book_text)
    if not chunks:
        raise ValueError("No chunks extracted from book text.")
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    if faiss is not None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, chunks, "faiss"
    # Fallback: keep raw matrix and do brute-force L2 in NumPy.
    return embeddings, chunks, "numpy"


def retrieve_chunks(
    question: str,
    index: Any,
    chunks: Sequence[Dict[str, Any]],
    embedder: SentenceTransformer,
    top_k: int,
) -> List[Dict[str, Any]]:
    q_vec = embedder.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    ).astype(np.float32)
    if faiss is not None and hasattr(index, "search"):
        distances, indices = index.search(q_vec, top_k)
    else:
        # index is expected to be [num_chunks, dim] float32 matrix.
        # L2 distance (same metric as IndexFlatL2) for one query.
        d = np.sum((index - q_vec[0]) ** 2, axis=1)
        k = min(max(1, top_k), len(d))
        idx_sorted = np.argsort(d)[:k]
        distances = np.expand_dims(d[idx_sorted], 0)
        indices = np.expand_dims(idx_sorted.astype(np.int64), 0)
    out: List[Dict[str, Any]] = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[int(idx)]
        out.append(
            {
                "rank": rank,
                "distance": float(dist),
                "chunk_id": c["chunk_id"],
                "label": c["label"],
                "chapter": c["chapter"],
                "paragraph": c["paragraph"],
                "text": c["text"],
            }
        )
    return out


def build_rag_prompt(question: str, retrieved: Sequence[Dict[str, Any]]) -> str:
    context = "\n\n".join([f"[{r['label']}]\n{r['text']}" for r in retrieved])
    return f"""# Episodic Memory Benchmark
You are participating in an episodic memory test based on retrieved text.
Answer using only the retrieved chunks. If unsure, say unsure.

## Retrieved Relevant Chunks:
{context}

## Question:
{question}

Return only the final answer."""


def build_dp_prompt(question: str, book_text: str, max_chars: Optional[int]) -> str:
    source = book_text
    if max_chars is not None and max_chars > 0:
        source = source[:max_chars]
    return f"""# Episodic Memory Benchmark
You are participating in an episodic memory test.
Answer using only the book text. If unsure, say unsure.

## Book Text:
{source}

## Question:
{question}

Return only the final answer."""


def build_chat_prompts(tokenizer: AutoTokenizer, user_prompts: Sequence[str]) -> List[str]:
    out: List[str] = []
    for user_prompt in user_prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        out.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return out


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parse_row_key(row: pd.Series, row_idx: int) -> str:
    if "combined_q_idx" in row and pd.notna(row["combined_q_idx"]):
        return f"combined_q_idx:{int(row['combined_q_idx'])}"
    return f"row_idx:{int(row_idx)}"


def as_int_or_none(v: Any) -> Optional[int]:
    if pd.isna(v):
        return None
    return int(v)


def as_str_or_none(v: Any) -> Optional[str]:
    if pd.isna(v):
        return None
    return str(v)


def default_jobs(repo_root: Path, preset: str) -> List[Dict[str, Path | str]]:
    if preset not in {"minerva150", "minimal150"}:
        raise ValueError(f"Unsupported preset: {preset}")

    qa_folder = (
        repo_root / "refactoring_files" / "qna_three_books" / f"combined_balanced_150_story_{'minerva_balanced' if preset == 'minerva150' else 'minimal'}"
    )
    return [
        {
            "name": "news",
            "book_txt": repo_root / "refactoring_files" / "artifacts" / "storybook" / "news_llm_retry" / "news_book.txt",
            "qa_csv": qa_folder / f"news_news_llm_retry_df_qa.combined_story_{'minerva_balanced' if preset == 'minerva150' else 'minimal'}_150.csv",
        },
        {
            "name": "scifi",
            "book_txt": repo_root / "refactoring_files" / "artifacts" / "storybook" / "20260312T170655Z" / "scifi_book.txt",
            "qa_csv": qa_folder / f"scifi_20260312T170655Z_df_qa.combined_story_{'minerva_balanced' if preset == 'minerva150' else 'minimal'}_150.csv",
        },
        {
            "name": "thriller",
            "book_txt": repo_root / "refactoring_files" / "artifacts" / "storybook" / "20260311T152054Z" / "book.txt",
            "qa_csv": qa_folder / f"thriller_20260311T152054Z_df_qa.combined_story_{'minerva_balanced' if preset == 'minerva150' else 'minimal'}_150.csv",
        },
    ]


def parse_job_arg(job_arg: str) -> Dict[str, Path | str]:
    # name|book_txt|qa_csv
    parts = [p.strip() for p in job_arg.split("|")]
    if len(parts) != 3:
        raise ValueError(f"Invalid --job format: {job_arg}")
    return {"name": parts[0], "book_txt": Path(parts[1]), "qa_csv": Path(parts[2])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-14B vLLM colocated runner for RAG and DP.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--modes", nargs="+", default=["dp"], choices=["rag", "dp"])
    parser.add_argument("--preset", choices=["minerva150", "minimal150"], default="minerva150")
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Optional custom job in format name|book_txt|qa_csv (can repeat). If omitted, --preset defaults are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("refactoring_files") / "Models_answers" / "Qwen2.5_14B_instruct",
    )
    parser.add_argument("--top-k-rag", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dp-max-book-chars", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    # vLLM tuning knobs
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def ensure_dependencies(modes: Sequence[str]) -> None:
    missing: List[str] = []
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")
    if torch is None:
        missing.append("torch")
    if "rag" in modes and SentenceTransformer is None:
        missing.append("sentence-transformers")
    if tqdm is None:
        missing.append("tqdm")
    if AutoTokenizer is None:
        missing.append("transformers")
    if LLM is None or SamplingParams is None:
        missing.append("vllm")

    if missing:
        uniq = sorted(set(missing))
        deps = " ".join(uniq)
        raise ModuleNotFoundError(
            "Missing required dependencies: "
            + ", ".join(uniq)
            + "\nInstall with:\n"
            + f"python -m pip install {deps}"
        )


def run_mode_for_job(
    job: Dict[str, Path | str],
    mode: str,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    llm: LLM,
    embedder: Optional[SentenceTransformer],
    model_id: str,
    output_dir: Path,
) -> None:
    name = str(job["name"])
    book_txt = Path(job["book_txt"])
    qa_csv = Path(job["qa_csv"])

    if not book_txt.exists():
        raise FileNotFoundError(f"Missing book text file: {book_txt}")
    if not qa_csv.exists():
        raise FileNotFoundError(f"Missing QA CSV file: {qa_csv}")

    out_jsonl = output_dir / f"{name}_qwen25_14b_{mode}.answers.jsonl"
    run_log = output_dir / f"{name}_qwen25_14b_{mode}.run.log"
    run_id = f"qwen25_14b_{mode}_{utc_run_tag()}"

    log_line(f"run_id={run_id}", run_log)
    log_line(f"mode={mode}", run_log)
    log_line(f"model_id={model_id}", run_log)
    log_line(f"book_txt={book_txt}", run_log)
    log_line(f"qa_csv={qa_csv}", run_log)
    log_line(f"output={out_jsonl}", run_log)

    book_text = book_txt.read_text(encoding="utf-8", errors="replace")
    df = pd.read_csv(qa_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    done = set()
    if args.resume and out_jsonl.exists():
        for r in read_jsonl(out_jsonl):
            if r.get("status") == "ok" and r.get("row_key"):
                done.add(r["row_key"])

    log_line(f"resume_done={len(done)} rows_total={len(df)}", run_log)

    rag_index = None
    rag_chunks: Optional[List[Dict[str, Any]]] = None
    if mode == "rag":
        if embedder is None:
            raise RuntimeError("RAG mode requested but sentence-transformers embedder is not initialized.")
        log_line(f"building_rag_index embed_model={args.embed_model}", run_log)
        rag_index, rag_chunks, rag_backend = build_rag_index(book_text, embedder)
        log_line(f"rag_backend={rag_backend}", run_log)
        log_line(f"chunks_indexed={len(rag_chunks)}", run_log)

    pending_rows: List[Tuple[int, pd.Series, str]] = []
    for i in range(len(df)):
        row = df.iloc[i]
        rk = parse_row_key(row, i)
        if rk in done:
            continue
        pending_rows.append((i, row, rk))

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    progress = tqdm(total=len(pending_rows), desc=f"{name}:{mode}")
    for batch in batched(pending_rows, max(1, args.batch_size)):
        prompts: List[str] = []
        meta: List[Dict[str, Any]] = []

        for i, row, rk in batch:
            question = str(row["question"])
            retrieved = None
            if mode == "rag":
                assert rag_index is not None and rag_chunks is not None
                retrieved = retrieve_chunks(question, rag_index, rag_chunks, embedder, args.top_k_rag)
                user_prompt = build_rag_prompt(question, retrieved)
            else:
                user_prompt = build_dp_prompt(question, book_text, args.dp_max_book_chars)

            prompts.append(user_prompt)
            meta.append(
                {
                    "row_idx": int(i),
                    "row_key": rk,
                    "q_idx": as_int_or_none(row["q_idx"]) if "q_idx" in row else int(i),
                    "combined_q_idx": as_int_or_none(row["combined_q_idx"]) if "combined_q_idx" in row else None,
                    "question": question,
                    "retrieval_type": as_str_or_none(row["retrieval_type"]) if "retrieval_type" in row else None,
                    "get": as_str_or_none(row["get"]) if "get" in row else None,
                    "correct_answer": as_str_or_none(row["correct_answer"]) if "correct_answer" in row else None,
                    "retrieved": retrieved,
                }
            )

        rendered = build_chat_prompts(tokenizer, prompts)

        try:
            outputs = llm.generate(rendered, sampling_params, use_tqdm=False)
            for m, out in zip(meta, outputs):
                answer = (out.outputs[0].text or "").strip() if out.outputs else ""
                rec = {
                    "run_id": run_id,
                    "timestamp_utc": utc_now(),
                    "status": "ok",
                    "error": None,
                    "mode": mode,
                    "model_id": model_id,
                    "row_key": m["row_key"],
                    "row_idx": m["row_idx"],
                    "q_idx": m["q_idx"],
                    "combined_q_idx": m["combined_q_idx"],
                    "question": m["question"],
                    "retrieval_type": m["retrieval_type"],
                    "get": m["get"],
                    "correct_answer": m["correct_answer"],
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "answer": answer,
                    "top_k": args.top_k_rag if mode == "rag" else None,
                    "retrieved_chunks": m["retrieved"] if mode == "rag" else None,
                }
                append_jsonl(out_jsonl, rec)
                done.add(m["row_key"])
                progress.update(1)
        except Exception as batch_exc:  # noqa: BLE001
            # Fallback row-wise to isolate failing prompts while preserving progress.
            err_text = f"{type(batch_exc).__name__}: {batch_exc}"
            log_line(f"batch_error size={len(batch)} err={err_text}", run_log)
            for rendered_prompt, m in zip(rendered, meta):
                status = "ok"
                error = None
                answer = ""
                try:
                    out = llm.generate([rendered_prompt], sampling_params, use_tqdm=False)[0]
                    answer = (out.outputs[0].text or "").strip() if out.outputs else ""
                except Exception as one_exc:  # noqa: BLE001
                    status = "error"
                    error = f"{type(one_exc).__name__}: {one_exc}"

                rec = {
                    "run_id": run_id,
                    "timestamp_utc": utc_now(),
                    "status": status,
                    "error": error,
                    "mode": mode,
                    "model_id": model_id,
                    "row_key": m["row_key"],
                    "row_idx": m["row_idx"],
                    "q_idx": m["q_idx"],
                    "combined_q_idx": m["combined_q_idx"],
                    "question": m["question"],
                    "retrieval_type": m["retrieval_type"],
                    "get": m["get"],
                    "correct_answer": m["correct_answer"],
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "answer": answer,
                    "top_k": args.top_k_rag if mode == "rag" else None,
                    "retrieved_chunks": m["retrieved"] if mode == "rag" else None,
                }
                append_jsonl(out_jsonl, rec)
                if status == "ok":
                    done.add(m["row_key"])
                progress.update(1)

    progress.close()
    log_line(f"run_done mode={mode} name={name} done_ok={len(done)}", run_log)


def main() -> None:
    args = parse_args()
    ensure_dependencies(args.modes)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    custom_jobs = [parse_job_arg(j) for j in args.job]
    jobs = custom_jobs if custom_jobs else default_jobs(repo_root, args.preset)

    for j in jobs:
        j["book_txt"] = (repo_root / j["book_txt"]).resolve() if isinstance(j["book_txt"], Path) and not j["book_txt"].is_absolute() else Path(j["book_txt"]).resolve()
        j["qa_csv"] = (repo_root / j["qa_csv"]).resolve() if isinstance(j["qa_csv"], Path) and not j["qa_csv"].is_absolute() else Path(j["qa_csv"]).resolve()

    print(f"[{utc_now()}] CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[{utc_now()}] GPU={torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model_id,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )

    embedder: Optional[SentenceTransformer] = None
    if "rag" in args.modes:
        embedder = SentenceTransformer(args.embed_model, device="cpu")

    run_manifest = {
        "timestamp_utc": utc_now(),
        "model_id": args.model_id,
        "embed_model": args.embed_model,
        "output_dir": str(output_dir),
        "modes": list(args.modes),
        "jobs": [
            {"name": j["name"], "book_txt": str(j["book_txt"]), "qa_csv": str(j["qa_csv"])}
            for j in jobs
        ],
        "settings": {
            "top_k_rag": args.top_k_rag,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "dp_max_book_chars": args.dp_max_book_chars,
            "batch_size": args.batch_size,
            "max_rows": args.max_rows,
            "resume": args.resume,
            "seed": args.seed,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        },
    }
    manifest_path = output_dir / f"qwen25_14b_vllm_colocate_manifest_{utc_run_tag()}.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"[{utc_now()}] manifest={manifest_path}")

    started = time.time()
    for job in jobs:
        for mode in args.modes:
            run_mode_for_job(
                job=job,
                mode=mode,
                args=args,
                tokenizer=tokenizer,
                llm=llm,
                embedder=embedder,
                model_id=args.model_id,
                output_dir=output_dir,
            )
    elapsed = time.time() - started
    print(f"[{utc_now()}] done elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
