import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

# Import models
import models.model_logistic_regression as lr
import models.model_naive_bayes as nb
import models.model_linear_svm as svm
import models.model_lstm as lstm

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

# Load datasets
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
    df['category'] = label  # Note: using 'category' as in model files
    data.append(df)

df = pd.concat(data, ignore_index=True)

# Build text column
df = build_feature_column(df)

# Assuming column name is 'text'
X = df['text']
y = df['category']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train & evaluate
models = {
    "Logistic Regression": lr,
    "Naive Bayes": nb,
    "SVM": svm,
    # "LSTM": lstm
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.train(X_train, y_train)  # each model must have train()

    preds = model.batch_predict(X_test)  # batch prediction

    acc = accuracy_score(y_test, preds)
    results[name] = acc

# Display results
print("\n===== Accuracy Comparison =====")
for k, v in results.items():
    print(f"{k}: {v:.4f}")


import json

with open("model_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to model_results.json")