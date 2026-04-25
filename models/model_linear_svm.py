
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)


DATA_DIR = "."
SAVE_DIR = "."
CLASSES  = ["business", "education", "entertainment", "sports", "technology"]

DATA_FILES = {
    "business":      "business_data.csv",
    "education":     "education_data.csv",
    "entertainment": "entertainment_data.csv",
    "technology":    "technology_data.csv",
    "sports":        "sports_data.csv",
}



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
    text = re.sub(r"<[^>]+>",       " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]",      " ", text)
    text = re.sub(r"\s+",           " ", text).strip()
    return text

def build_feature_column(df):
    df = df.copy()
    df["text"] = (
        df["headlines"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["content"].fillna("")
    ).apply(clean_text)
    return df



def split_data(df):
    X, y = df["text"], df["category"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────────────────────
# 4. BUILD PIPELINE
# ──────────────────────────────────────────────────────────────

def build_pipeline(calibrated=False):
    tfidf = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)

    base_pipe = Pipeline([("tfidf", tfidf), ("clf", svm)])

    if calibrated:
        # CalibratedClassifierCV wraps the whole pipeline
        return CalibratedClassifierCV(base_pipe, cv=3)
    return base_pipe



def train_and_evaluate(pipe, X_train, X_val, X_test, y_train, y_val, y_test):
    pipe.fit(X_train, y_train)

    val_preds  = pipe.predict(X_val)
    test_preds = pipe.predict(X_test)

    val_acc  = accuracy_score(y_val,  val_preds)
    val_f1   = f1_score(y_val,  val_preds,  average="macro")
    test_acc = accuracy_score(y_test, test_preds)
    test_f1  = f1_score(y_test, test_preds, average="macro")

    print("\n" + "═" * 50)
    print("  Linear SVM — Results")
    print("═" * 50)
    print(f"  Val  Accuracy : {val_acc:.4f}   Val  Macro-F1 : {val_f1:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}   Test Macro-F1 : {test_f1:.4f}")
    print(f"\n{classification_report(y_test, test_preds)}")

    return pipe, test_preds, {
        "val_acc": val_acc, "val_f1": val_f1,
        "test_acc": test_acc, "test_f1": test_f1,
    }



def plot_confusion_matrix(test_preds, y_test, metrics, save_dir=SAVE_DIR):
    cm = confusion_matrix(y_test, test_preds, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=ax, linewidths=0.5)
    ax.set_title(
        f"Linear SVM — Confusion Matrix\n"
        f"Test Acc: {metrics['test_acc']:.3f}  Macro-F1: {metrics['test_f1']:.3f}"
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=30); plt.tight_layout()
    path = os.path.join(save_dir, "svm_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_top_features(pipe, save_dir=SAVE_DIR, n_top=15):
    # Handle both plain pipeline and calibrated wrapper
    if hasattr(pipe, "named_steps"):
        tfidf = pipe.named_steps["tfidf"]
        clf   = pipe.named_steps["clf"]
    else:
        inner = pipe.estimator
        tfidf = inner.named_steps["tfidf"]
        clf   = inner.named_steps["clf"]

    vocab = np.array(tfidf.get_feature_names_out())
    coef  = clf.coef_   # shape: (n_classes, n_features)

    colors_map = {
        "business":      "#378ADD", "education":     "#1D9E75",
        "entertainment": "#D85A30", "sports":        "#7F77DD",
        "technology":    "#BA7517",
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    for ax, (i, cls) in zip(axes, enumerate(clf.classes_)):
        top_idx    = np.argsort(coef[i])[-n_top:][::-1]
        top_words  = vocab[top_idx]
        top_scores = coef[i][top_idx]
        ax.barh(top_words[::-1], top_scores[::-1],
                color=colors_map.get(cls, "gray"))
        ax.set_title(cls, fontsize=11, fontweight="bold")
        ax.set_xlabel("SVM coefficient")
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle(f"Linear SVM — Top {n_top} features per class", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "svm_top_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def main():
    print("═" * 55)
    print("  Linear SVM — News Classifier")
    print("═" * 55)

    df = load_data()
    df = build_feature_column(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    pipe = build_pipeline(calibrated=False)
    pipe, test_preds, metrics = train_and_evaluate(
        pipe, X_train, X_val, X_test, y_train, y_val, y_test)

    plot_confusion_matrix(test_preds, y_test, metrics)
    plot_top_features(pipe)

    return pipe


if __name__ == "__main__":
    main()