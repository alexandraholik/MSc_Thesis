# Detecting Contextomized Quotes in English News Headlines

**Alexandra Holíková** · MSc Thesis · University of Amsterdam  
`alexandra.holikova@student.uva.nl` · Student ID: 14236788

---

## Overview

This repository contains all code, data processing pipelines, and evaluation notebooks for an MSc thesis on **contextomy detection in English news headlines**. The work extends Song et al. (2023) — who introduced the task and a supervised contrastive learning framework (QuoteCSE) for Korean — to English, using zero-shot and few-shot methods that require no labelled training data.

**Core task:** Given a news article whose headline contains a direct quotation, predict whether that quote is *contextomized* (taken out of context to distort meaning) or *modified* (faithfully representing the speaker's intent in the body text).

---

## Repository Structure

```
MSc_Thesis/
├── 01_data_loading_inspection.ipynb     # Data collection from CC-News
├── 02_quote_matching.ipynb              # Headline-to-body quote matching
├── 03_annotation_sampling.ipynb         # Stratified sampling for annotation
├── 04_detection.ipynb                   # Zero-/few-shot detection (English)
├── 05_korean_evaluation.ipynb           # Replication on Korean test set
├── 05_korean_evaluation_fewshot.ipynb   # Few-shot variants on Korean test set
├── 06_error_analysis.ipynb              # Error analysis across methods/languages
├── data/                                # Processed datasets (see below)
└── Thesis_4.json                        # Zotero bibliography (CSL-JSON)
```

---

## Notebooks

### `01_data_loading_inspection.ipynb` — Data Collection
Streams and filters the CC-News corpus (`stanford-oval/ccnews` via HuggingFace Datasets). Scans up to 500,000 articles and selects those meeting three eligibility criteria: (i) headline contains a quoted span of at least a minimum word count, (ii) body text has at least 50 words, (iii) source is a recognised general/political news outlet. Outputs `data/cc_news_full.parquet`.

### `02_quote_matching.ipynb` — Quote Matching
For each eligible article, embeds the headline quote and every body sentence using `paraphrase-multilingual-mpnet-base-v2` (Sentence-BERT). Selects the highest-similarity body sentence as the candidate match and records a three-sentence context window (previous + match + next). Outputs `data/cc_news_matched.parquet`.

### `03_annotation_sampling.ipynb` — Annotation Sampling
Draws a stratified sample from the matched articles, stratified by similarity tercile (`low` / `mid` / `high`) following the verbatim-exclusion protocol of Song et al. (2023). Exports a CSV formatted for hand-annotation in Google Sheets.

### `04_detection.ipynb` — Zero-Shot Detection (English)
Benchmarks seven approaches on the 100-article hand-annotated English test set. Primary metric: AUC (target: Song et al.'s QuoteCSE AUC = 0.768 on Korean, supervised). Approaches:

| # | Method | Model |
|---|--------|-------|
| 1 | Cosine similarity threshold (τ = 0.5) | `paraphrase-multilingual-mpnet-base-v2` |
| 2 | NLI-based detection | `mDeBERTa-v3-base-mnli-xnli` |
| 3 | GPT-4o zero-shot, context window | GPT-4o |
| 4 | GPT-4o chain-of-thought, context window | GPT-4o |
| 5 | GPT-4o zero-shot, full article | GPT-4o |
| 6 | GPT-4o chain-of-thought, full article | GPT-4o |
| 7 | GPT-4o few-shot | GPT-4o |

### `05_korean_evaluation.ipynb` / `05_korean_evaluation_fewshot.ipynb` — Korean Evaluation
Applies the same detection approaches to the 1,600-article Korean test set from Song et al. (2023) (814 contextomized, 786 modified). Includes additional LLM variants: a Korean-language prompt (`cot_ko`) and a cross-lingual chain-of-thought variant (`cot_xlcot`).

### `06_error_analysis.ipynb` — Error Analysis
Analyses where methods agree and disagree across both test sets. Splits samples into *easy* (unanimous agreement) and *hard* (methods disagree) following the framework of Larooij & Graus (2026). Produces pairwise agreement matrices and qualitative analysis of failure cases.

---

## Data

The `data/` directory is not tracked in this repository due to size and licensing constraints. The following files are expected:

| File | Description | Produced by |
|------|-------------|-------------|
| `data/cc_news_full.parquet` | Filtered CC-News snapshot | Notebook 01 |
| `data/cc_news_matched.parquet` | Articles with matched body sentences | Notebook 02 |
| `data/annotation_sample.csv` | Stratified sample for annotation | Notebook 03 |
| `data/test_set_predictions.csv` | English test set with all method predictions | Notebook 04 |
| `data/nli_scores.csv` | Pre-computed NLI scores (mDeBERTa) | Notebook 04 |
| `data/korean_matched.csv` | Korean test set with matched sentences | Notebook 05 |
| `data/korean_predictions.csv` | Korean test set with all method predictions | Notebook 05 |

The raw Korean annotated dataset is from Song et al. (2023). Access it at the [original repository](https://github.com/s-kosugi/contextomized-quote-detection) or by contacting the authors.

---

## Setup

### Requirements

```bash
pip install datasets pandas numpy scikit-learn matplotlib sentence-transformers pysbd transformers torch openai
```

### Environment variables

```bash
export OPENAI_API_KEY="your-key-here"   # required for Notebooks 04–05 (GPT-4o approaches)
```

### Running order

Notebooks are numbered sequentially and must be run in order. Each notebook reads from files produced by the previous one.

```
01 → 02 → 03 → [annotation] → 04 → 05 → 06
```

Manual annotation is required between Notebook 03 and Notebook 04. The annotation guidelines follow the contextomy definition in McGlone (2005) and the labelling scheme of Song et al. (2023).

---

## Reference

Song, J., Song, S., Park, S., Han, J., & Cha, M. (2023). Detecting contextomized quotes in news headlines by contrastive learning. *Findings of EACL 2023*, 698–710.

---

## Citation

If you use this code or data, please cite the thesis once published (details to be added).

---

## License

Code in this repository is released under the MIT License. The CC-News corpus is subject to [Common Crawl's terms of use](https://commoncrawl.org/terms-of-use). The Korean dataset is subject to the licensing terms of Song et al. (2023).
