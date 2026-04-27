import pandas as pd
import re
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# -------------------------
# CLEANING
# -------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_feature_column(df):
    df = df.copy()
    df["text"] = (
        df["headlines"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["content"].fillna("")
    ).apply(clean_text)
    return df

# -------------------------
# LOAD DATA
# -------------------------
files = [
    ("data/business_data.csv", "business"),
    ("data/sports_data.csv", "sports"),
    ("data/technology_data.csv", "technology"),
    ("data/education_data.csv", "education"),
    ("data/entertainment_data.csv", "entertainment")
]

data = []
for file, label in files:
    df = pd.read_csv(file)
    df["category"] = label
    data.append(df)

df = pd.concat(data, ignore_index=True)
df = build_feature_column(df)

X = df["text"]
y = df["category"]

# -------------------------
# TRAIN
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ DEFINE MODELS HERE (NO IMPORT)
models = {
    "lr": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "nb": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", MultinomialNB())
    ]),
    "svm": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LinearSVC())
    ])
}

os.makedirs("saved_models", exist_ok=True)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    with open(f"saved_models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

print("✅ Models saved successfully!")