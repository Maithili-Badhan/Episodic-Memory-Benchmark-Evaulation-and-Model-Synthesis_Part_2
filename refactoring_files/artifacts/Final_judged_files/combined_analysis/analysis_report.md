# Final Episodic Memory Benchmark Analysis

## Coverage
- Total evaluated rows: 3597
- Model-book-method conditions: 24

## Leaderboard
- Best model by mean F1: **Llama3.1_8B_instruct** (0.4793, EM 0.4578)
- Lowest model by mean F1: **Qwen3_4B_Instruct** (0.4455, EM 0.4381)

## Method Effect
- Global mean F1 (RAG): 0.4884
- Global mean F1 (DP): 0.4328
- Net delta (RAG - DP): +0.0557

## Best/Worst Conditions
- Best condition: **SCIFI_RAG** (mean F1 0.5137)
- Hardest condition: **THRILLER_DP** (mean F1 0.3965)

## Category Behavior
- Easiest category (avg over all models/methods): **Match and Compare** (0.9355)
- Hardest category (avg over all models/methods): **Spot the Differences** (0.0526)

## DP vs RAG (Estimation)
- See Gardner-Altman estimation plots for model-level and book-level effects.

### Mean Delta F1 (DP - RAG) By Model
- Llama3.1_8B_instruct: -0.0346 [-0.0803, +0.0096] (paired n=450)
- Qwen3_4B_Instruct: -0.0574 [-0.0832, -0.0337] (paired n=447)
- Qwen2.5_14B_Instruct: -0.0579 [-0.0974, -0.0203] (paired n=450)
- Phi3_mini_4K_instruct: -0.0744 [-0.1161, -0.0313] (paired n=450)

### Mean Delta F1 (DP - RAG) By Book
- SCIFI: -0.0530 [-0.0887, -0.0167] (paired n=600)
- NEWS: -0.0531 [-0.0856, -0.0217] (paired n=597)
- THRILLER: -0.0621 [-0.0933, -0.0309] (paired n=600)

## Data Notes
- Qwen3_4B_Instruct / NEWS_RAG has 147 rows (expected 150).
