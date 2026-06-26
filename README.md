# Automatic Contextomy Detection in News Headlines

Code for the MSc thesis *Automatic Contextomy Detection in News Headlines* (Alexandra Holíková, University of Amsterdam, 2026).

The project evaluates LLM-as-a-judge methods for detecting contextomised headline quotes — cases where a speaker's quoted words are reproduced accurately but stripped of context that changes their meaning. The primary evaluation uses the Korean benchmark from Song et al. (2023) (n = 1,600); an English demonstration set is constructed from scratch using CC-News (n = 100 annotated).

The central finding is that input unit selection drives performance: when GPT-4o receives the matched body quote directly (quote-to-quote condition), it reaches AUC 0.781 on the Korean benchmark, exceeding the supervised QuoteCSE baseline (AUC 0.768) without any task-specific training.

---

## Repository structure

```
.
├── data/                          # Korean pipeline data (original matching)
│   ├── korean_matched.csv         # Sentence-level matching output
│   ├── korean_predictions.csv     # ctx / full / language variant scores
│   ├── korean_gpt_direct_q2q.csv  # GPT-4o Direct q2q scores
│   ├── korean_gpt_cot_q2q.csv     # GPT-4o CoT q2q scores
│   └── contextomized_quote.pkl    # Song et al. (2023) source data
│
├── data_new/                      # New pipeline data (English + Korean q2q)
│   ├── cc_news_full.csv           # Raw CC-News filtered articles
│   ├── english_annotation_sample_v2.csv  # 2,000-article annotation pool
│   ├── english_hand_annotated.csv # 100 labelled English articles
│   ├── english_matched.csv        # English matching output (sent + q2q)
│   ├── english_predictions.csv    # All English detection scores
│   ├── korean_matched.csv         # Korean quote-level matching output
│   ├── korean_predictions_v2.csv  # All Korean scores incl. q2q conditions
│   ├── english_error_analysis.csv # Per-sample group flags (English)
│   └── korean_error_analysis.csv  # Per-sample group flags (Korean)
│
├── figures/                       # Saved plots (PDF/PNG)
│
├── 01_english_set_construction.ipynb
├── 02_matching.ipynb
├── 04_detection_english.ipynb
├── 05_korean_evaluation.ipynb
├── 05b_korean_q2q.ipynb
└── 06_error_analysis_new.ipynb
```

---

## Notebooks

### `01_english_set_construction.ipynb` — English set construction

Builds the English annotation pool from CC-News.

1. Streams up to 800,000 articles from `stanford-oval/ccnews`; keeps English articles (language score ≥ 0.9) with non-empty headline and body.
2. Extracts headline quotes (quotation marks, ≥ 4 words) and body quotes (quotation marks + attribution verb within 80 characters, ≥ 4 words).
3. Excludes near-verbatim pairs: headline quote is a substring of a body quote, or vice versa, or word-overlap ratio > 0.9.
4. Draws a flat random sample of 2,000 articles for manual annotation.

**Output:** `data_new/english_annotation_sample_v2.csv`

---

### `02_matching.ipynb` — Headline-to-body matching

Matches each headline quote to the most semantically similar span in the article body. Runs on the manually annotated English set; the same procedure is applied to the Korean set inside notebook 05.

- **Model:** `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers)
- **Two candidate spaces per article:**
  - Body quotes (direct speech with attribution verb)
  - Body sentences (all sentences)
- For each space, selects the highest cosine-similarity match and records a margin score (gap to the second-best candidate).
- Stores `best_body_quote` / `best_quote_sim` (q2q condition) and `best_body_sentence` / `prev_sentence` / `next_sentence` / `best_sim` (ctx condition).
- Includes a manual correction block for four articles where automatic extraction missed the relevant span.

**Input:** `data_new/english_hand_annotated.csv`  
**Output:** `data_new/english_matched.csv`

---

### `04_detection_english.ipynb` — English detection experiments

Evaluates all detection methods on the 100-article English test set (10 contextomised / 90 modified).

**Methods evaluated:**

| # | Method | Input condition |
|---|--------|----------------|
| 1 | Cosine similarity baseline | ctx (sentence) |
| 2 | Cosine similarity baseline | q2q (body quote) |
| 3 | GPT-4o Direct | ctx |
| 4 | GPT-4o Direct | q2q |
| 5 | GPT-4o Direct | full article |
| 6 | GPT-4o CoT | ctx |
| 7 | GPT-4o CoT | q2q |
| 8 | GPT-4o CoT | full article |

All GPT calls use `gpt-4o` at `temperature=0`. The Direct prompt asks for a binary label and probability; the CoT prompt adds two intermediate reasoning steps before the label. Bootstrapped 95% CIs (10,000 resamples) are reported for all AUC estimates.

**Output:** `data_new/english_predictions.csv`

---

### `05_korean_evaluation.ipynb` — Korean detection experiments (ctx / full / language variants)

Applies detection methods to the Song et al. (2023) Korean benchmark (n = 1,600; 814 contextomised / 786 modified).

Loads the Korean matched data from `data_new/korean_matched.csv` and joins the full article body from the original Song et al. pickle for full-article variants.

**Methods evaluated:**

| # | Method | Input | Prompt language |
|---|--------|-------|----------------|
| 1 | Cosine similarity | sentence (ctx) | — |
| 2 | mDeBERTa NLI | ctx | — |
| 3 | GPT-4o Direct | ctx | English |
| 4 | GPT-4o CoT | ctx | English |
| 5 | GPT-4o Direct | full | English |
| 6 | GPT-4o CoT | full | English |
| 7 | GPT-4o CoT | ctx | Korean |
| 8 | GPT-4o CoT | ctx | XL-CoT (English reasoning on Korean input) |

Previously computed scores for methods 3–8 are loaded from saved CSVs; only new API runs are executed. Results are saved to `data/korean_predictions.csv`.

---

### `05b_korean_q2q.ipynb` — Korean q2q conditions

Adds the quote-to-quote input condition to the Korean evaluation. Loads all ctx/full/language results from `data/korean_predictions.csv` and merges q2q matching columns from `data_new/korean_matched.csv`.

**New conditions:**

| Method | Input |
|--------|-------|
| Cosine | q2q |
| GPT-4o Direct | q2q |
| GPT-4o CoT | q2q |

Produces the combined predictions file used by all downstream analysis and the final results table.

**Output:** `data/korean_predictions_v2.csv` (all conditions merged)

> **Note on directory split:** `data/korean_matched.csv` contains sentence-level matching (ctx input); `data_new/korean_matched.csv` contains quote-level matching (q2q input). They are different files with different content. `data/korean_predictions_v2.csv` is the canonical file for all Korean results.

---

### `06_error_analysis_new.ipynb` — Error analysis

Partitions samples by cross-method agreement and looks for patterns separating easy from hard cases. Covers both the English and Korean sets.

**Three groups per sample:**
- **easy_correct** — all six LLM methods agree and are right
- **easy_wrong** — all six methods agree and are wrong
- **hard** — methods are split

**Analyses:**
- Vote-count distribution (U-shape = high agreement; mass in middle = frequent disagreement)
- Pairwise prediction agreement matrix for both datasets
- Per-method accuracy on easy vs. hard subsets
- Feature comparison across groups: `best_quote_sim`, `best_sim`, quote length, context length
- Qualitative inspection of unanimous errors and most-split hard cases, with Korean→English translation via MyMemory API

**Inputs:** `data_new/english_predictions.csv`, `data_new/korean_predictions_v2.csv`  
**Outputs:** `data_new/english_error_analysis.csv`, `data_new/korean_error_analysis.csv`

---

## Setup

```bash
pip install pandas numpy scikit-learn sentence-transformers openai python-dotenv datasets matplotlib seaborn requests
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

The notebooks expect the following directory structure to already exist, or will create it:

```
data/        # Korean pipeline (Song et al. source data required)
data_new/    # New pipeline (created by notebook 01)
figures/     # Created by notebooks 04, 05b, 06
```

The Song et al. (2023) Korean dataset (`data/contextomized_quote.pkl`) is not included in this repository. See [Song et al. (2023)](https://aclanthology.org/2023.findings-eacl.52/) for access.

---

## Data flow

```
CC-News (streaming)
    └── 01_english_set_construction
            └── data_new/english_annotation_sample_v2.csv
                    └── [manual annotation]
                            └── data_new/english_hand_annotated.csv
                                    └── 02_matching
                                            └── data_new/english_matched.csv
                                                    └── 04_detection_english
                                                            └── data_new/english_predictions.csv

Song et al. (2023) Korean data
    └── [quote-level matching, separate script]
            └── data_new/korean_matched.csv  (q2q input)
            └── data/korean_matched.csv      (ctx input)
                    └── 05_korean_evaluation
                            └── data/korean_predictions.csv
                                    └── 05b_korean_q2q
                                            └── data/korean_predictions_v2.csv

data_new/english_predictions.csv  ──┐
data/korean_predictions_v2.csv    ──┴── 06_error_analysis_new
```

---

## Citation

If you use this code, please cite:

```
Holíková, A. (2026). Automatic Contextomy Detection in News Headlines.
MSc Thesis, University of Amsterdam.
```

Song et al. (2023) benchmark:

```
Song, S., et al. (2023). Contextomized Quote Detection in News Headlines.
Findings of EACL 2023.
```
