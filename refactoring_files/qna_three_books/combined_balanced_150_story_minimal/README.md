# Episodic Memory Benchmark: Final Optimized Setup

This is the final setup for our 3-book benchmark using the balanced story QA sets (150 questions per book).

## Dataset

Use these 3 files as the benchmark inputs:

- `news_news_llm_retry_df_qa.combined_story_minimal_150.csv`
- `scifi_20260312T170655Z_df_qa.combined_story_minimal_150.csv`
- `thriller_20260311T152054Z_df_qa.combined_story_minimal_150.csv`

Total questions: `450` (150 per book).

## Target Models (Answering Models)

Run all 3:

1. `Qwen2.5-14B-Instruct`
2. `Mistral-Nemo-Instruct-2407` (12B)
3. `gemma-2-9b-it`

## Memory Strategies (Must Include All 3)

1. **In-Context**
- Give full book text + question.
- No retrieval.
- This is the direct prompting baseline.

2. **RAG**
- Chunk by paragraph.
- Embedding model (open-source): `BAAI/bge-m3` (or `intfloat/e5-large-v2`).
- Retrieve top `k=10` chunks.
- Provide retrieved chunks + question.

3. **Fine-Tuning**
- Use QLoRA (4-bit) for each of the 3 models.
- Train only on **single-event QA** style first (`n_items_correct_answer == 1`) for stability.
- Evaluate on full benchmark (all categories).

## Judge Model (Open-Source, Not GPT)

Default judge:

- `Qwen/Qwen2.5-72B-Instruct`

Why:
- strong instruction-following
- good structured JSON outputs
- much better judge quality than small local models

Fallback judge (if needed):
- `meta-llama/Llama-3.3-70B-Instruct`

## Recommended Hyperparameters

Use the same generation settings for fairness:

- `temperature=0.0`
- `top_p=1.0`
- `max_new_tokens=256`
- fixed `seed`

### RAG
- chunk type: paragraph
- top_k: `10`
- optional rerank: top 10 -> top 5

### Fine-Tuning (QLoRA)
- quantization: `4-bit nf4`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- learning rate: `1e-4`
- scheduler: `cosine`
- warmup ratio: `0.03`
- effective batch size: `16`
- max seq length: `2048`

Epochs (optimized for your setup/data size):
- `Qwen2.5-14B-Instruct`: `6`
- `Mistral-Nemo-Instruct-2407`: `6`
- `gemma-2-9b-it`: `8`

## Evaluation Metrics

1. **Primary:** lenient F1 (same logic as paper)
2. **Secondary:** Kendall's tau for chronological ordering questions
3. **Statistical tests:** Wilcoxon signed-rank with Holm correction

## Run Matrix

Per book, run:

- 3 models x 3 memory strategies = `9` runs per book

Across all 3 books:

- `27` total runs

## Output Format (Recommended)

For each run, save:

- model name
- memory strategy
- book id
- question
- prediction
- identified items (judge)
- matching scores (judge)
- F1
- (optional) tau for chronology

Store in:

- `artifacts/eval_final/<model>/<strategy>/<book>.jsonl`
- `artifacts/eval_final/summary_metrics.csv`

## Practical Compute Plan

If GPU is limited (Colab free tier), do in this order:

1. Run all **In-Context** and **RAG** first for all 3 models.
2. Then run **Fine-Tuning** model by model.
3. Start with `gemma-2-9b-it` FT first (fastest), then Mistral, then Qwen.

## Notes for Fair Comparison

- Keep the same QA set and same decoding params for all methods.
- In-Context must not use retriever outputs.
- RAG must use only retrieved chunks, not full book text.
- Fine-tuned inference must not include full book at test time.
- Keep judge prompt fixed for all runs.

This setup gives a clean, fair comparison of `In-Context vs RAG vs Fine-Tuning` on your balanced episodic-memory benchmark, with an open-source high-quality judge model.
