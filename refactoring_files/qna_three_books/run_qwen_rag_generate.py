#!/usr/bin/env python
"""
Run RAG-only answer generation for episodic-memory QA CSV files.

No judge/evaluation is performed here.
Outputs are saved as JSONL logs for later scoring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import faiss
except ModuleNotFoundError:
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
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None


SYSTEM_PROMPT = "You are an expert in memory tests."


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def split_book_into_paragraph_chunks(book_content: str) -> List[Dict]:
    """
    Split by chapter then paragraph as requested in the official RAG template.
    If no chapter delimiter is found, fallback to global paragraph splitting.
    """
    chunks: List[Dict] = []
    raw_chapters = book_content.split("Chapter ")

    if len(raw_chapters) > 1:
        for chapter_idx, chapter in enumerate(raw_chapters[1:], start=1):
            paragraphs = [p.strip() for p in chapter.strip().split("\n\n") if p.strip()]
            for para_idx, para in enumerate(paragraphs, start=1):
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "text": para,
                        "label": f"Chapter {chapter_idx}, Paragraph {para_idx}",
                        "chapter": chapter_idx,
                        "paragraph": para_idx,
                    }
                )
    else:
        paragraphs = [p.strip() for p in book_content.split("\n\n") if p.strip()]
        for para_idx, para in enumerate(paragraphs, start=1):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": para,
                    "label": f"Paragraph {para_idx}",
                    "chapter": None,
                    "paragraph": para_idx,
                }
            )
    return chunks


def build_rag_index(book_content: str, embedding_model_name: str) -> Tuple[faiss.IndexFlatL2, List[Dict], SentenceTransformer]:
    """
    Build FAISS L2 index from paragraph chunks.
    """
    chunks = split_book_into_paragraph_chunks(book_content)
    if not chunks:
        raise ValueError("No chunks found in book content.")

    embed_model = SentenceTransformer(embedding_model_name)
    embeddings = embed_model.encode(
        [c["text"] for c in chunks],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks, embed_model


def retrieve_chunks(
    question: str,
    index: faiss.IndexFlatL2,
    chunks: List[Dict],
    embed_model: SentenceTransformer,
    top_k: int,
) -> List[Dict]:
    q_emb = embed_model.encode(
        [question],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    distances, indices = index.search(q_emb, top_k)

    retrieved: List[Dict] = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        item = dict(chunks[idx])
        item["rank"] = rank
        item["distance"] = float(dist)
        retrieved.append(item)
    return retrieved


def clip_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_stop > int(max_chars * 0.5):
        return cut[: last_stop + 1]
    return cut + "..."


def build_user_prompt(
    question: str,
    retrieved: List[Dict],
    prompt_style: str,
    max_chars_per_chunk: Optional[int],
) -> str:
    context = "\n\n".join(
        [f"[{c['label']}]\n{clip_text(c['text'], max_chars_per_chunk)}" for c in retrieved]
    )
    if prompt_style == "compact":
        return f"""# Episodic Memory Benchmark
Use only the retrieved text below to answer the question.
If the answer is not in the retrieved text, say you are unsure.
Return only the final answer, with no extra explanation.

## Retrieved Chunks:
{context}

## Question:
{question}
"""
    return f"""# Episodic Memory Benchmark
You are participating in an episodic memory test, based on the data below, which was
retrieved from a book. You need to read it and internalize as if you had personally
experienced the events described. After the text, you will find a question about the
content. Please answer this question based solely on the information provided.

## Retrieved Relevant Chunks from the Book:
{context}

## Question:
{question}

Please answer the question to the best of your ability, based only on the information
provided in the relevant chunks above. If you are unsure, it's okay to say so. Do not
invent or assume information not explicitly stated."""


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def preflight_models(base_url: str, timeout_sec: int) -> List[str]:
    """
    Verify that the OpenAI-compatible server is reachable and return available model ids.
    """
    url = normalize_base_url(base_url) + "/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to {url}. "
            "Start llama-server first, then retry.\n"
            "Example:\n"
            ".\\llama.cpp\\build\\bin\\Release\\llama-server.exe "
            "-m .\\models\\qwen25\\Qwen2.5-14B-Instruct-Q4_K_M.gguf "
            "--host 127.0.0.1 --port 8080 -c 8192"
        ) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unexpected /models response from {url}: {payload[:300]}") from exc

    models = []
    for item in data.get("data", []):
        model_id = item.get("id")
        if model_id:
            models.append(str(model_id))
    return models


def call_model(
    client: OpenAI,
    model_name: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    request_retries: int,
    request_retry_backoff_sec: float,
) -> str:
    last_exc = None
    retries = max(1, request_retries)
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            sleep_sec = request_retry_backoff_sec * (2 ** (attempt - 1))
            time.sleep(sleep_sec)
    raise last_exc


def infer_default_book_path(qa_csv_path: Path, repo_root: Path) -> Optional[Path]:
    """
    Auto-map known QA files to their corresponding book.txt files.
    """
    name = qa_csv_path.name.lower()
    storybook_root = repo_root / "refactoring_files" / "artifacts" / "storybook"

    mapping = {
        "news_news_llm_retry_df_qa.combined_story_minimal_150.csv": storybook_root / "news_llm_retry" / "book.txt",
        "scifi_20260312t170655z_df_qa.combined_story_minimal_150.csv": storybook_root / "20260312T170655Z" / "book.txt",
        "thriller_20260311t152054z_df_qa.combined_story_minimal_150.csv": storybook_root / "20260311T152054Z" / "book.txt",
    }
    return mapping.get(name)


def build_row_key(row: pd.Series, row_idx: int) -> str:
    if "combined_q_idx" in row and pd.notna(row["combined_q_idx"]):
        return f"combined_q_idx:{int(row['combined_q_idx'])}"
    return f"row_idx:{int(row_idx)}"


def load_existing_done_keys(output_jsonl: Path) -> set:
    if not output_jsonl.exists():
        return set()
    done = set()
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if str(row.get("status", "ok")).lower() != "ok":
                    continue
                key = row.get("row_key")
                if key:
                    done.add(str(key))
                    continue
                if row.get("combined_q_idx") is not None:
                    done.add(f"combined_q_idx:{int(row['combined_q_idx'])}")
            except Exception:
                continue
    return done


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG generation with local Qwen GGUF server (OpenAI-compatible).")
    parser.add_argument("--qa-csv", required=True, help="Path to QA CSV file.")
    parser.add_argument("--book-txt", default=None, help="Path to book.txt. If omitted, auto-mapping is attempted.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path for generated answers/logs.")
    parser.add_argument("--progress-log", default=None, help="Optional plain-text progress log path.")

    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-chars-per-chunk", type=int, default=None)
    parser.add_argument("--prompt-style", choices=["detailed", "compact"], default="detailed")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed rows in output JSONL.")

    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--llm-api-key", default="local")
    parser.add_argument("--llm-model", default="qwen2.5-14b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--request-retry-backoff-sec", type=float, default=2.0)
    return parser.parse_args()


def log_line(msg: str, progress_log: Optional[Path]) -> None:
    print(msg, flush=True)
    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        with progress_log.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def main() -> int:
    args = parse_args()

    missing = []
    if faiss is None:
        missing.append("faiss-cpu")
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")
    if OpenAI is None:
        missing.append("openai")
    if SentenceTransformer is None:
        missing.append("sentence-transformers")

    if missing:
        print(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + "\nInstall with: pip install "
            + " ".join(sorted(set(missing))),
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parents[2]

    qa_csv_path = Path(args.qa_csv).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    progress_log = Path(args.progress_log).resolve() if args.progress_log else None

    if args.book_txt:
        book_txt_path = Path(args.book_txt).resolve()
    else:
        guessed = infer_default_book_path(qa_csv_path, repo_root)
        if guessed is None:
            raise ValueError(
                f"Could not auto-map QA CSV '{qa_csv_path.name}'. "
                "Please provide --book-txt explicitly."
            )
        book_txt_path = guessed.resolve()

    if not qa_csv_path.exists():
        raise FileNotFoundError(f"QA CSV not found: {qa_csv_path}")
    if not book_txt_path.exists():
        raise FileNotFoundError(f"Book text not found: {book_txt_path}")

    run_id = f"qwen_rag_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    log_line(f"[{utc_now_iso()}] run_id={run_id}", progress_log)
    log_line(f"[{utc_now_iso()}] qa_csv={qa_csv_path}", progress_log)
    log_line(f"[{utc_now_iso()}] book_txt={book_txt_path}", progress_log)
    log_line(f"[{utc_now_iso()}] llm_base_url={args.llm_base_url}", progress_log)

    try:
        served_models = preflight_models(args.llm_base_url, args.timeout_sec)
    except Exception as exc:  # noqa: BLE001
        log_line(f"[{utc_now_iso()}] server_preflight ERROR {exc}", progress_log)
        return 1

    if served_models:
        log_line(f"[{utc_now_iso()}] server_models={served_models}", progress_log)
        if args.llm_model not in served_models:
            log_line(
                f"[{utc_now_iso()}] WARNING requested model '{args.llm_model}' "
                f"not in served ids. Use one of: {served_models}",
                progress_log,
            )

    df = pd.read_csv(qa_csv_path)
    if "question" not in df.columns:
        raise ValueError("QA CSV must contain a 'question' column.")

    if args.max_questions is not None:
        df = df.head(args.max_questions).copy()
        log_line(f"[{utc_now_iso()}] max_questions={args.max_questions}", progress_log)

    done_row_keys = set()
    if args.resume:
        done_row_keys = load_existing_done_keys(output_jsonl)
        log_line(f"[{utc_now_iso()}] resume=True already_done={len(done_row_keys)}", progress_log)

    book_content = book_txt_path.read_text(encoding="utf-8", errors="replace")
    log_line(f"[{utc_now_iso()}] building RAG index using {args.embedding_model}", progress_log)
    index, chunks, embed_model = build_rag_index(book_content, args.embedding_model)
    log_line(f"[{utc_now_iso()}] chunks_indexed={len(chunks)}", progress_log)

    client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key, timeout=args.timeout_sec)

    total = len(df)
    generated = 0
    skipped = 0
    errors = 0

    for i, row in df.iterrows():
        row_key = build_row_key(row, i)
        q_idx = int(row["q_idx"]) if "q_idx" in row and pd.notna(row["q_idx"]) else int(i)
        if row_key in done_row_keys:
            skipped += 1
            continue

        question = str(row["question"])
        retrieved = retrieve_chunks(question, index, chunks, embed_model, args.top_k)
        user_prompt = build_user_prompt(
            question=question,
            retrieved=retrieved,
            prompt_style=args.prompt_style,
            max_chars_per_chunk=args.max_chars_per_chunk,
        )

        log_line(
            f"[{utc_now_iso()}] q={i+1}/{total} row_key={row_key} q_idx={q_idx} retrieving={len(retrieved)}",
            progress_log,
        )

        answer_text = ""
        status = "ok"
        error_msg = None
        try:
            answer_text = call_model(
                client=client,
                model_name=args.llm_model,
                user_prompt=user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                request_retries=args.request_retries,
                request_retry_backoff_sec=args.request_retry_backoff_sec,
            )
        except Exception as exc:
            status = "error"
            error_msg = f"{type(exc).__name__}: {exc}"
            errors += 1
            log_line(f"[{utc_now_iso()}] q_idx={q_idx} ERROR {error_msg}", progress_log)

        out_row = {
            "run_id": run_id,
            "timestamp_utc": utc_now_iso(),
            "status": status,
            "error": error_msg,
            "qa_csv": str(qa_csv_path),
            "book_txt": str(book_txt_path),
            "row_key": row_key,
            "row_idx": int(i),
            "q_idx": q_idx,
            "combined_q_idx": int(row["combined_q_idx"]) if "combined_q_idx" in row and pd.notna(row["combined_q_idx"]) else None,
            "question": question,
            "cue": str(row["cue"]) if "cue" in row and pd.notna(row["cue"]) else None,
            "retrieval_type": str(row["retrieval_type"]) if "retrieval_type" in row and pd.notna(row["retrieval_type"]) else None,
            "get": str(row["get"]) if "get" in row and pd.notna(row["get"]) else None,
            "correct_answer": str(row["correct_answer"]) if "correct_answer" in row and pd.notna(row["correct_answer"]) else None,
            "llm_model": args.llm_model,
            "llm_base_url": args.llm_base_url,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "rag_embedding_model": args.embedding_model,
            "rag_top_k": args.top_k,
            "prompt_style": args.prompt_style,
            "max_chars_per_chunk": args.max_chars_per_chunk,
            "retrieved_chunks": [
                {
                    "rank": c["rank"],
                    "chunk_id": c["chunk_id"],
                    "label": c["label"],
                    "chapter": c["chapter"],
                    "paragraph": c["paragraph"],
                    "distance": c["distance"],
                    "text": c["text"],
                }
                for c in retrieved
            ],
            "answer": answer_text,
        }
        append_jsonl(output_jsonl, out_row)
        generated += 1
        if status == "ok":
            done_row_keys.add(row_key)

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    log_line(
        f"[{utc_now_iso()}] done generated={generated} skipped={skipped} errors={errors} output={output_jsonl}",
        progress_log,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
