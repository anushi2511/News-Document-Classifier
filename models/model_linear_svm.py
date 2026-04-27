import pickle
import os
import re

MODEL_PATH = "saved_models/svm.pkl"

_model = None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise Exception("Model not trained. Run train_models.py first.")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

def predict(text):
    model = load_model()
    text = clean_text(text)
    return model.predict([text])[0]