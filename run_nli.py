import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, f1_score, classification_report

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
NLI_MODEL  = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
BATCH_SIZE = 32   # increase to 64 if GPU has >16GB VRAM
THRESHOLD  = 0.5
INPUT_CSV  = 'data/test_set_100.csv'
OUTPUT_CSV = 'data/nli_scores.csv'

print(f"Device: {DEVICE}")

# --- Load data ---
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=['label']).copy()
df['label'] = df['label'].astype(int)
for col in ['prev_sentence', 'next_sentence']:
    df[col] = df[col].fillna('')

def build_premise(row):
    parts = [row['prev_sentence'], row['best_body_sentence'], row['next_sentence']]
    return ' '.join(p for p in parts if p).strip()

premises   = [build_premise(row) for _, row in df.iterrows()]
hypotheses = df['headline_quote'].tolist()
y_true     = df['label'].values

# --- Load model ---
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
model.eval()

print(f"Label order: {model.config.id2label}")
CONTRADICTION_IDX = 0  # update if id2label shows contradiction is not index 0

# --- Batched inference ---
nli_scores = []
n_batches  = -(-len(premises) // BATCH_SIZE)  # ceiling division

for i in range(0, len(premises), BATCH_SIZE):
    batch_p = premises[i:i+BATCH_SIZE]
    batch_h = hypotheses[i:i+BATCH_SIZE]
    inputs  = tokenizer(
        batch_p, batch_h,
        return_tensors='pt', truncation=True,
        max_length=512, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
    nli_scores.extend(probs[:, CONTRADICTION_IDX].tolist())
    print(f"  Batch {i//BATCH_SIZE + 1}/{n_batches} done")

nli_scores = np.array(nli_scores)

# --- Evaluate ---
preds = (nli_scores >= THRESHOLD).astype(int)
auc   = roc_auc_score(y_true, nli_scores)
f1    = f1_score(y_true, preds, zero_division=0)

print(f"\nAUC : {auc:.3f}  |  F1 (τ={THRESHOLD}) : {f1:.3f}")
print(classification_report(y_true, preds, target_names=['modified', 'contextomized']))

# --- Save scores for notebook ---
df['score_nli'] = nli_scores
df['pred_nli']  = preds
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
