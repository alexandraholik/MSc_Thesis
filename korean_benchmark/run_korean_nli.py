import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
NLI_MODEL  = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
BATCH_SIZE = 32
THRESHOLD  = 0.5
INPUT_CSV  = 'data/korean_matched.csv'
OUTPUT_CSV = 'data/korean_nli_scores.csv'

print(f"Device: {DEVICE}")

# --- Load data ---
df = pd.read_csv(INPUT_CSV)
df['label'] = df['label'].astype(int)
for col in ['prev_sentence', 'next_sentence', 'best_body_sentence']:
    df[col] = df[col].fillna('')

y_true = df['label'].values
print(f"Loaded {len(df)} articles")

def build_premise(row):
    parts = [row['prev_sentence'], row['best_body_sentence'], row['next_sentence']]
    return ' '.join(p for p in parts if p).strip()

premises   = [build_premise(row) for _, row in df.iterrows()]
hypotheses = df['headline_quote'].tolist()

# --- Load model ---
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
model.eval()

print(f"Label order: {model.config.id2label}")
CONTRADICTION_IDX = 2  # confirmed: {0: entailment, 1: neutral, 2: contradiction}

# --- Batched inference ---
nli_scores = []
n_batches  = -(-len(premises) // BATCH_SIZE)

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
preds       = (nli_scores >= THRESHOLD).astype(int)
auc         = roc_auc_score(y_true, nli_scores)
f1          = f1_score(y_true, preds, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision   = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\nAUC: {auc:.3f}  |  F1: {f1:.3f}  |  Sens: {sensitivity:.3f}  |  Spec: {specificity:.3f}  |  Prec: {precision:.3f}")
print(classification_report(y_true, preds, target_names=['modified', 'contextomized']))

# --- Save ---
df['score_nli'] = nli_scores
df['pred_nli']  = preds
df[['headline_quote', 'label', 'score_nli', 'pred_nli']].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
