# Episodic Memory Benchmark for Small Language Models

This repository contains the dataset, evaluation pipeline, and analysis code for our study on episodic memory in small, locally executable language models.

## Overview

Large Language Models often exhibit strong semantic recall but struggle with episodic memory — the ability to recall events grounded in time, location, and entity context.  
This project introduces a controlled benchmark designed to isolate and evaluate episodic memory behavior under constrained inference settings.
<img width="1083" height="334" alt="CogAI drawio" src="https://github.com/user-attachments/assets/a007ede3-a7e4-4e4e-a3fe-22f3cc154033" />
The benchmark is inspired by the Minerva capability taxonomy and focuses on narrative-level memory rather than surface retrieval.

## Dataset

- Source narrative: 49-chapter fictional novel
- Preprocessed into episodic narrative blocks
- Each episode implicitly represents:
  - Temporal position
  - Location
  - Participating entities
  - Event content

### Question Sets

- 3 independent question sets
- 40 questions per set (120 total)
- Balanced across:
  - Atomic memory tasks
  - Composite episodic reasoning
  - Minerva capability categories

Each question is annotated with:
- Capability category
- Cue specification
- Reasoning rationale

## Evaluation Methods

We evaluate two prompting strategies:

1. **Direct Prompting**
2. **Retrieval-Augmented Generation (RAG)** using TF-IDF chunk retrieval

Evaluation metrics include:
- Token-level precision, recall, and F1
- LLM-as-a-judge correctness scoring
- Chronological awareness
- Answer length and variance analysis

## Results Summary

- Overall episodic recall remains low across all configurations
![Quantitative_Analysis](https://github.com/user-attachments/assets/3081b535-c1d9-4dab-a733-cad3910f0585)
- RAG provides limited and inconsistent improvements
  ![Gardner_Altman_for_qset2](https://github.com/user-attachments/assets/b38adf2d-499f-472c-aecc-623c0e2255bd)
- Performance correlates with event density and entity recurrence
  ![Event_Analysis_Chart](https://github.com/user-attachments/assets/11c1927c-597c-4954-93bd-0f5253b27c4e)
- Retrieval alone does not induce stable episodic reasoning

## Reproducibility

All experiments are designed to run under local execution constraints.
No proprietary APIs are required.

## Citation

If you use this work, please cite our paper: Coming soon...

@article{underway}
