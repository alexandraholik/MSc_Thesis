"""
Korean quote-to-quote matching.
Matches each headline quote against the body quotes provided by Song et al.,
NOT against all body sentences. This makes the candidate pool identical to
QuoteCSE's and produces a like-for-like comparison.
"""
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_PKL  = 'data/contextomized_quote.pkl'
OUTPUT_CSV = 'data/korean_matched.csv'
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# --- Load data ---
with open(INPUT_PKL, 'rb') as f:
    df = pickle.load(f)

df = df.dropna(subset=['label']).copy()
df['label'] = df['label'].astype(int)

print(f"Loaded {len(df)} labelled articles")
print(f"Mean body quotes per article: {df['body_quotes'].apply(len).mean():.1f}")

# --- Load model ---
embedder = SentenceTransformer(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")


def match_headline_to_body_quotes(headline_quote, body_quotes):
    """Match headline quote against extracted body quote spans.
    Returns best body quote, its neighbours in the list, cosine sim, and margin."""
    if not headline_quote or not body_quotes:
        return '', '', '', 0.0, 0.0

    hq = headline_quote[0] if isinstance(headline_quote, list) else headline_quote

    hq_emb = embedder.encode([hq], normalize_embeddings=True)
    bq_emb = embedder.encode(body_quotes, normalize_embeddings=True, batch_size=64)
    sims = cosine_similarity(hq_emb, bq_emb)[0]

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    # margin to second-best
    if len(sims) > 1:
        sorted_sims = np.sort(sims)[::-1]
        margin = float(sorted_sims[0] - sorted_sims[1])
    else:
        margin = best_sim

    # neighbours in the body_quotes list (not sentence neighbours)
    prev_quote = body_quotes[best_idx - 1] if best_idx > 0 else ''
    next_quote = body_quotes[best_idx + 1] if best_idx < len(body_quotes) - 1 else ''

    return body_quotes[best_idx], prev_quote, next_quote, best_sim, margin


# --- Run matching ---
best_quotes, prev_quotes, next_quotes, best_sims, margins = [], [], [], [], []

for i, (_, row) in enumerate(df.iterrows()):
    best, prev, nxt, sim, margin = match_headline_to_body_quotes(
        row['headline_quote'], row['body_quotes']
    )
    best_quotes.append(best)
    prev_quotes.append(prev)
    next_quotes.append(nxt)
    best_sims.append(sim)
    margins.append(margin)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)} matched")

df['best_body_sentence'] = best_quotes  # keep column name for downstream compat
df['prev_sentence']      = prev_quotes
df['next_sentence']      = next_quotes
df['best_sim']           = best_sims
df['sim_margin']         = margins

# Drop columns not needed downstream
df_out = df.drop(columns=['body', 'body_quotes', 'headline'])
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Mean best_sim: {np.mean(best_sims):.3f}")
print(f"Median best_sim: {np.median(best_sims):.3f}")
