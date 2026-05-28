import pickle
import pandas as pd
import numpy as np
import re
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
df['body']  = df['body'].fillna('')

print(f"Loaded {len(df)} articles")

# --- Load model ---
embedder = SentenceTransformer(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")

def split_sentences(text):
    """Strip HTML and split on sentence boundaries."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def match_headline_to_body(headline_quote, body_text):
    """Return best-matching body sentence, its neighbours, and cosine score."""
    sentences = split_sentences(body_text)
    if not sentences:
        return '', '', '', 0.0

    hq_emb   = embedder.encode([headline_quote], normalize_embeddings=True)
    body_emb = embedder.encode(sentences, normalize_embeddings=True, batch_size=64)
    sims     = cosine_similarity(hq_emb, body_emb)[0]
    best_idx = int(np.argmax(sims))

    prev_sent = sentences[best_idx - 1] if best_idx > 0 else ''
    next_sent = sentences[best_idx + 1] if best_idx < len(sentences) - 1 else ''

    return sentences[best_idx], prev_sent, next_sent, float(sims[best_idx])

# --- Run matching ---
best_sentences, prev_sentences, next_sentences, best_sims = [], [], [], []

for i, (_, row) in enumerate(df.iterrows()):
    best, prev, nxt, sim = match_headline_to_body(
        str(row['headline_quote']), str(row['body'])
    )
    best_sentences.append(best)
    prev_sentences.append(prev)
    next_sentences.append(nxt)
    best_sims.append(sim)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)} matched")

df['best_body_sentence'] = best_sentences
df['prev_sentence']      = prev_sentences
df['next_sentence']      = next_sentences
df['best_sim']           = best_sims

# Drop full body text — not needed downstream
df_out = df.drop(columns=['body'])
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"Mean similarity: {np.mean(best_sims):.3f}")
