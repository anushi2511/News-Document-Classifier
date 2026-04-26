"""
News Document Classifier — Model 4: LSTM with PyTorch
======================================================
Role       : Neural model — learns its own word embeddings from your data,
             no pretrained weights of any kind.
Expected   : ~92-94% macro-F1
Dependencies: torch, numpy, pandas, scikit-learn (metrics only),
              matplotlib, seaborn

Architecture:
  Token IDs → Embedding layer (trained from random init)
              ↓
           LSTM cell (single layer, batch_first=True)
              ↓
         Last hidden state → Dense (5 classes) → (CrossEntropyLoss uses logits)

Why LSTM beats MLP on TF-IDF for text:
  - TF-IDF destroys word order. LSTM reads the sequence left-to-right and
    maintains a hidden state that carries contextual information forward.
  - The embedding layer projects sparse one-hot token IDs into a dense
    continuous space — it learns that "goal" and "score" are semantically
    close in the sports domain entirely from your CSV data.
  - No pretrained weights; every parameter is initialised randomly and
    updated via BPTT (Backpropagation Through Time).

Training notes:
  - Autograd handles BPTT automatically via PyTorch's computation graph.
  - Gradient clipping (norm threshold 5.0) prevents exploding gradients.
  - Adam optimiser with per-parameter adaptive learning rates.
  - Sequences longer than MAX_SEQ are truncated; shorter ones are
    zero-padded on the left.
  - GPU/MPS acceleration is used automatically when available.

Runtime: ~2-5 min on CPU for the default config (much faster than NumPy).
         Reduce EMBED_DIM / HIDDEN_DIM or EPOCHS to speed up further.

Run:
    python model_lstm.py
"""

import os, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

DATA_DIR  = "data"
SAVE_DIR  = "."
CLASSES   = ["business", "education", "entertainment", "sports", "technology"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

# Model hyperparameters
VOCAB_SIZE = 15_000   # top-k words kept
MAX_SEQ    = 200      # tokens per document (truncate / pad)
EMBED_DIM  = 64       # embedding vector size
HIDDEN_DIM = 128      # LSTM hidden state size
EPOCHS     = 10
BATCH_SIZE = 64
LR         = 1e-3     # Adam learning rate
CLIP_NORM  = 5.0      # gradient clipping threshold

DATA_FILES = {
    "business":      "business_data.csv",
    "education":     "education_data.csv",
    "entertainment": "entertainment_data.csv",
    "technology":    "technology_data.csv",
    "sports":        "sports_data.csv",
}

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device selection: CUDA > MPS (Apple Silicon) > CPU
DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available()  else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")


# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────────

def load_data(data_dir=DATA_DIR):
    dfs = []
    for category, filename in DATA_FILES.items():
        path = os.path.join(data_dir, filename)
        df   = pd.read_csv(path, on_bad_lines="skip", engine="python")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(combined)} samples")
    print(combined["category"].value_counts().to_string())
    return combined

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>",        " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]",       " ", text)
    text = re.sub(r"\s+",            " ", text).strip()
    return text

def build_feature_column(df):
    df = df.copy()
    df["text"] = (
        df["headlines"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["content"].fillna("")
    ).apply(clean_text)
    return df


# ──────────────────────────────────────────────────────────────
# 2. VOCABULARY & TOKENISATION
# ──────────────────────────────────────────────────────────────

def build_vocab(texts, vocab_size=VOCAB_SIZE):
    """Count word frequencies and keep the top vocab_size words."""
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    # token 0 = <PAD>, token 1 = <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def tokenise(texts, vocab, max_seq=MAX_SEQ):
    """Convert list of strings → padded integer array (N, max_seq)."""
    unk = vocab["<UNK>"]
    pad = vocab["<PAD>"]
    out = []
    for text in texts:
        ids = [vocab.get(w, unk) for w in text.split()]
        ids = ids[:max_seq]                          # truncate
        ids = [pad] * (max_seq - len(ids)) + ids     # left-pad
        out.append(ids)
    return np.array(out, dtype=np.int64)


# ──────────────────────────────────────────────────────────────
# 3. PYTORCH DATASET
# ──────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    """Wraps tokenised integer arrays and labels for DataLoader."""

    def __init__(self, X_ids: np.ndarray, y: np.ndarray):
        # Store as tensors on CPU; DataLoader will batch them
        self.X = torch.tensor(X_ids, dtype=torch.long)
        self.y = torch.tensor(y,     dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────
# 4. PYTORCH LSTM MODEL
# ──────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Single-layer LSTM classifier built with PyTorch nn.Module.

    Parameters
    ----------
    vocab_size  : size of the vocabulary (including <PAD> and <UNK>)
    embed_dim   : dimensionality of the word embedding table
    hidden_dim  : number of LSTM hidden units
    num_classes : number of output classes (5 for this project)
    pad_idx     : index of the <PAD> token (embeddings set to zero,
                  gradients not propagated through padding positions)
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_classes, pad_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embedding layer — equivalent to the original self.emb table
        self.embedding = nn.Embedding(
            vocab_size, embed_dim,
            padding_idx=pad_idx      # keeps <PAD> vector as zero
        )

        # Single-layer LSTM — replaces the hand-written LSTM cell
        # batch_first=True → input/output shape: (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Output (dense) layer — equivalent to W_out / b_out
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x : (batch_size, seq_len) — token ID tensor

        Returns: logits of shape (batch_size, num_classes)
                 (raw scores; loss fn applies softmax internally)
        """
        # (batch, seq, embed_dim)
        embedded = self.embedding(x)

        # lstm_out : (batch, seq, hidden_dim)
        # h_n      : (num_layers, batch, hidden_dim) — last hidden state
        # c_n      : (num_layers, batch, hidden_dim) — last cell state
        _, (h_n, _) = self.lstm(embedded)

        # Take the last layer's hidden state → (batch, hidden_dim)
        h_last = h_n[-1]

        # Project to class logits → (batch, num_classes)
        logits = self.fc(h_last)
        return logits


# ──────────────────────────────────────────────────────────────
# 5. TRAINING & EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser, device, clip_norm):
    """Run one full pass over the training DataLoader."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad()

        logits = model(X_batch)                     # forward pass
        loss   = criterion(logits, y_batch)         # cross-entropy

        loss.backward()                             # BPTT via autograd

        # Gradient clipping — same threshold as the NumPy version
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimiser.step()
        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_all(model, loader, device):
    """Return numpy arrays of predicted class indices."""
    model.eval()
    all_preds = []

    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)

    return np.concatenate(all_preds)


def fit(model, X_train_ids, y_train, X_val_ids=None, y_val=None,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        clip_norm=CLIP_NORM, device=DEVICE):
    """
    Train the model and return per-epoch losses and validation F1 scores.
    Mirrors the interface of the original LSTMClassifier.fit().
    """
    train_ds = NewsDataset(X_train_ids, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, drop_last=False)

    val_dl = None
    if X_val_ids is not None:
        val_ds = NewsDataset(X_val_ids,
                             y_val if y_val is not None
                             else np.zeros(len(X_val_ids), dtype=np.int64))
        val_dl = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    losses   = []
    val_f1s  = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        avg_loss = train_one_epoch(
            model, train_dl, criterion, optimiser, device, clip_norm
        )
        losses.append(avg_loss)

        vf1_str = ""
        if val_dl is not None and y_val is not None:
            val_preds = predict_all(model, val_dl, device)
            vf1 = f1_score(y_val, val_preds, average="macro")
            val_f1s.append(vf1)
            vf1_str = f"  Val Macro-F1: {vf1:.4f}"

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{epochs}  "
              f"Loss: {avg_loss:.4f}{vf1_str}  ({elapsed:.0f}s)")

    return losses, val_f1s


# ──────────────────────────────────────────────────────────────
# 6. TRAIN / VAL / TEST SPLIT
# ──────────────────────────────────────────────────────────────

def split_data(df):
    X, y = df["text"], df["category"].map(CLASS2IDX)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return (X_train.reset_index(drop=True),
            X_val.reset_index(drop=True),
            X_test.reset_index(drop=True),
            y_train.values, y_val.values, y_test.values)


# ──────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ──────────────────────────────────────────────────────────────

def plot_training_curves(losses, val_f1s, save_dir=SAVE_DIR):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, marker="o", color="#378ADD")
    ax1.set_title("Training Loss per Epoch"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss"); ax1.grid(alpha=0.3)

    if val_f1s:
        ax2.plot(val_f1s, marker="o", color="#1D9E75")
        ax2.set_title("Validation Macro-F1 per Epoch")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "lstm_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(preds, y_test, metrics, save_dir=SAVE_DIR):
    pred_labels = [IDX2CLASS[p] for p in preds]
    true_labels = [IDX2CLASS[t] for t in y_test]
    cm = confusion_matrix(true_labels, pred_labels, labels=CLASSES)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=ax, linewidths=0.5)
    ax.set_title(
        f"LSTM (PyTorch) — Confusion Matrix\n"
        f"Test Acc: {metrics['test_acc']:.3f}  Macro-F1: {metrics['test_f1']:.3f}"
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=30); plt.tight_layout()
    path = os.path.join(save_dir, "lstm_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("═" * 55)
    print("  LSTM (PyTorch) — News Classifier")
    print("═" * 55)

    # Load & preprocess
    df = load_data()
    df = build_feature_column(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Vocabulary (built on training text only — no data leakage)
    vocab = build_vocab(X_train, vocab_size=VOCAB_SIZE)

    # Tokenise
    X_train_ids = tokenise(X_train, vocab)
    X_val_ids   = tokenise(X_val,   vocab)
    X_test_ids  = tokenise(X_test,  vocab)

    model = LSTMClassifier(
        vocab_size  = len(vocab),
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_classes = len(CLASSES),
        pad_idx     = vocab["<PAD>"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters:")
    print(f"  Embedding : {len(vocab)} × {EMBED_DIM} = {len(vocab)*EMBED_DIM:,}")
    print(f"  LSTM W_x  : 4×{HIDDEN_DIM} × {EMBED_DIM}  = {4*HIDDEN_DIM*EMBED_DIM:,}")
    print(f"  LSTM W_h  : 4×{HIDDEN_DIM} × {HIDDEN_DIM}  = {4*HIDDEN_DIM*HIDDEN_DIM:,}")
    print(f"  Output    : {len(CLASSES)} × {HIDDEN_DIM}  = {len(CLASSES)*HIDDEN_DIM:,}")
    print(f"  Total     : {total_params:,} parameters\n")

    print("Training LSTM...")
    losses, val_f1s = fit(
        model,
        X_train_ids, y_train,
        X_val_ids=X_val_ids, y_val=y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
    )

    test_ds    = NewsDataset(X_test_ids, y_test)
    test_dl    = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_preds = predict_all(model, test_dl, DEVICE)

    test_acc = accuracy_score(y_test, test_preds)
    test_f1  = f1_score(y_test, test_preds, average="macro")
    metrics  = {"test_acc": test_acc, "test_f1": test_f1}

    pred_labels = [IDX2CLASS[p] for p in test_preds]
    true_labels = [IDX2CLASS[t] for t in y_test]

    print("\n" + "═" * 50)
    print("  LSTM — Test Results")
    print("═" * 50)
    print(f"  Test Accuracy : {test_acc:.4f}   Test Macro-F1 : {test_f1:.4f}")
    print(f"\n{classification_report(true_labels, pred_labels)}")

    plot_training_curves(losses, val_f1s)
    plot_confusion_matrix(test_preds, y_test, metrics)

    return model, vocab


if __name__ == "__main__":
    main()

# ===============================
# 🔹 GLOBALS
# ===============================
_model = None
_vocab = None


def train(X_train=None, y_train=None):
    """
    Train model for external use
    """
    global _model, _vocab

    df = load_data()
    df = build_feature_column(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    _vocab = build_vocab(X_train)

    X_train_ids = tokenise(X_train, _vocab)
    X_val_ids   = tokenise(X_val, _vocab)

    model = LSTMClassifier(
        vocab_size=len(_vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(CLASSES),
        pad_idx=_vocab["<PAD>"],
    ).to(DEVICE)

    fit(
        model,
        X_train_ids, y_train,
        X_val_ids=X_val_ids, y_val=y_val,
    )

    _model = model
    return _model


def predict(text):
    """
    Predict single text (for Streamlit)
    """
    global _model, _vocab

    if _model is None or _vocab is None:
        train()

    text = clean_text(text)
    ids  = tokenise([text], _vocab)

    tensor = torch.tensor(ids, dtype=torch.long).to(DEVICE)

    _model.eval()
    with torch.no_grad():
        logits = _model(tensor)
        pred   = logits.argmax(dim=1).item()

    return IDX2CLASS[pred]


def batch_predict(texts):
    """
    Batch prediction (for comparison script)
    """
    global _model, _vocab

    if _model is None or _vocab is None:
        train()

    texts = [clean_text(t) for t in texts]
    ids   = tokenise(texts, _vocab)

    tensor = torch.tensor(ids, dtype=torch.long).to(DEVICE)

    _model.eval()
    with torch.no_grad():
        logits = _model(tensor)
        preds  = logits.argmax(dim=1).cpu().numpy()

    return [IDX2CLASS[p] for p in preds]