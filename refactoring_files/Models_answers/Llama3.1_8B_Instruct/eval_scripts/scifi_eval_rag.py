#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer


def to_jsonable(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return to_jsonable(v.item())
        except Exception:
            pass
    if hasattr(v, "tolist"):
        try:
            return to_jsonable(v.tolist())
        except Exception:
            pass
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [to_jsonable(x) for x in v]
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_qa(qa_file: Path) -> pd.DataFrame:
    suffix = qa_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(qa_file)
    if suffix == ".jsonl":
        return pd.read_json(qa_file, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(qa_file)
    raise ValueError(f"Unsupported QA format: {qa_file}")


def get_first(row: pd.Series, keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def split_chunks(text: str, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chapter = 0
    chunks = []
    for p_idx, para in enumerate(paras, start=1):
        m = re.match(r"^Chapter\s+(\d+)\b", para, flags=re.IGNORECASE)
        if m:
            chapter = int(m.group(1))

        if len(para) <= chunk_size:
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "label": f"Chapter {chapter if chapter else 0}, Paragraph {p_idx}",
                    "chapter": chapter if chapter else 0,
                    "paragraph": p_idx,
                    "text": para,
                }
            )
            continue

        step = max(1, chunk_size - overlap)
        for start in range(0, len(para), step):
            piece = para[start:start + chunk_size]
            if not piece:
                continue
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "label": f"Chapter {chapter if chapter else 0}, Paragraph {p_idx}",
                    "chapter": chapter if chapter else 0,
                    "paragraph": p_idx,
                    "text": piece,
                }
            )
            if start + chunk_size >= len(para):
                break
    return chunks


def l2_norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return a / n


def build_index(embedder: SentenceTransformer, chunks: list[dict[str, Any]]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    return l2_norm(embs.astype(np.float32))


def retrieve(embedder: SentenceTransformer, chunk_embs: np.ndarray, chunks: list[dict[str, Any]], question: str, top_k: int) -> list[dict[str, Any]]:
    q = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
    q = l2_norm(q)
    sims = (chunk_embs @ q.T).reshape(-1)
    k = min(top_k, len(chunks))
    idxs = np.argpartition(-sims, kth=k - 1)[:k]
    idxs = idxs[np.argsort(-sims[idxs])]

    out = []
    for rank, idx in enumerate(idxs, start=1):
        c = chunks[int(idx)]
        out.append(
            {
                "rank": rank,
                "chunk_id": c["chunk_id"],
                "label": c["label"],
                "chapter": c["chapter"],
                "paragraph": c["paragraph"],
                "distance": float(1.0 - float(sims[int(idx)])),
                "text": c["text"],
            }
        )
    return out


def build_prompt(question: str, retrieved: list[dict[str, Any]], prompt_style: str) -> str:
    ctx = "\n\n".join(f"[{c['label']}]\n{c['text']}" for c in retrieved)
    if prompt_style == "brief":
        return (
            "Answer only from retrieved context.\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Return only final answer."
        )
    return (
        "You are solving episodic-memory QA.\n"
        "Use only retrieved context. Follow query instruction exactly.\n"
        "If the query asks for list/set/intersection/difference/sorting, output only that result.\n"
        "No explanation.\n\n"
        f"RETRIEVED CONTEXT:\n{ctx}\n\n"
        f"QUESTION:\n{question}\n\n"
        "FINAL ANSWER:"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG QA evaluation for one qa file + one book file")
    p.add_argument("--qa-file", required=True)
    p.add_argument("--book-file", required=True)
    p.add_argument("--llamacpp-base-url", default="http://127.0.0.1:8081")
    p.add_argument("--api-key", default="not-needed")
    p.add_argument("--model-name", required=True)

    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--rag-top-k", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--prompt-style", choices=["brief", "detailed"], default="detailed")

    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max-questions", type=int, default=None)

    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep-seconds", type=float, default=0.0)

    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--run-name", default="")
    p.add_argument("--out-root", default="artifacts/storybooks")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    qa_path = Path(args.qa_file).resolve()
    book_path = Path(args.book_file).resolve()
    if not qa_path.is_file():
        raise FileNotFoundError(f"QA file missing: {qa_path}")
    if not book_path.is_file():
        raise FileNotFoundError(f"Book file missing: {book_path}")

    df = load_qa(qa_path)
    if "question" not in df.columns:
        raise SystemExit("QA file must contain question column")

    if args.end is not None:
        df = df.iloc[args.start:args.end]
    else:
        df = df.iloc[args.start:]
    if args.max_questions is not None:
        df = df.iloc[:args.max_questions]
    df = df.reset_index(drop=True)

    run_id = args.run_name or f"llama_rag_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    stem = qa_path.stem
    out_jsonl = out_root / f"{stem}.{run_id}.answers.jsonl"
    out_csv = out_root / f"{stem}.{run_id}.answers.csv"

    if args.overwrite:
        out_jsonl.unlink(missing_ok=True)
        out_csv.unlink(missing_ok=True)
    elif out_jsonl.exists():
        raise SystemExit(f"Output exists: {out_jsonl}. Use --overwrite")

    book_text = book_path.read_text(encoding="utf-8")
    chunks = split_chunks(book_text, args.chunk_size, args.chunk_overlap)
    embedder = SentenceTransformer(args.embedding_model)
    chunk_embs = build_index(embedder, chunks)

    client = OpenAI(base_url=normalize_base_url(args.llamacpp_base_url), api_key=args.api_key)

    print(f"Running RAG eval | rows={len(df)} | chunks={len(chunks)}")

    with out_jsonl.open("a", encoding="utf-8") as f:
        for row_idx, row in df.iterrows():
            q = str(row.get("question", "")).strip()
            q_idx = int(get_first(row, ["q_idx", "id"], row_idx))
            combined_q_idx = int(get_first(row, ["combined_q_idx"], row_idx))
            row_key = f"combined_q_idx:{combined_q_idx}"

            retrieved = retrieve(embedder, chunk_embs, chunks, q, args.rag_top_k)
            prompt = build_prompt(q, retrieved, args.prompt_style)

            status = "ok"
            err = None
            ans = ""
            t0 = time.time()
            try:
                out = client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise QA model."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                ans = (out.choices[0].message.content or "").strip()
            except Exception as e:
                status = "error"
                err = f"{type(e).__name__}: {e}"

            latency = round(time.time() - t0, 3)

            rec = {
                "run_id": run_id,
                "timestamp_utc": utc_now_iso(),
                "status": status,
                "error": err,
                "qa_csv": str(qa_path),
                "book_txt": str(book_path),
                "row_key": row_key,
                "row_idx": int(row_idx),
                "q_idx": q_idx,
                "combined_q_idx": combined_q_idx,
                "question": q,
                "cue": get_first(row, ["cue"]),
                "retrieval_type": get_first(row, ["retrieval_type"]),
                "get": get_first(row, ["get"]),
                "correct_answer": to_jsonable(get_first(row, ["correct_answer", "answer"])),
                "llm_model": args.model_name,
                "llm_base_url": normalize_base_url(args.llamacpp_base_url),
                "temperature": args.temperature,
                "max_new_tokens": args.max_tokens,
                "rag_embedding_model": args.embedding_model,
                "rag_top_k": args.rag_top_k,
                "prompt_style": args.prompt_style,
                "max_chars_per_chunk": args.chunk_size,
                "retrieved_chunks": retrieved,
                "answer": ans,
                "latency_sec": latency,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    rows = [json.loads(x) for x in out_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
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
        for r in rows
    ).to_csv(out_csv, index=False)

    print("Saved:", out_jsonl)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()