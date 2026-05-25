import json
import re
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


DATASET = Path("dataset") / "intents.json"
OUT_DIR = Path(".")


def ensure_nltk():
    # try to ensure minimal NLTK data; if unavailable, we'll fall back to simpler processing
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # lower
    text = text.lower()
    # remove punctuation and non-alphanumeric (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_lemmatize(text: str, lemmatizer, stop_words):
    # prefer nltk tokenizer if available; otherwise simple split
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    lemmas = []
    for t in tokens:
        try:
            lemmas.append(lemmatizer.lemmatize(t))
        except Exception:
            # fallback to porter stemmer
            lemmas.append(PorterStemmer().stem(t))
    return " ".join(lemmas)


def load_intents(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_dataset(intents_json: dict):
    pairs = []
    # intents_json expected to have top-level list or dict with 'intents'
    intents = intents_json.get("intents") if isinstance(intents_json, dict) else intents_json
    if isinstance(intents, dict):
        intents = intents.get("intents", [])
    for intent in intents:
        tag = intent.get("tag") or intent.get("intent") or intent.get("name")
        # training phrases could be under 'text' or 'patterns'
        texts = intent.get("text") or intent.get("patterns") or []
        for t in texts:
            if isinstance(t, dict):
                # sometimes entries are objects
                t = t.get("text") or t.get("pattern") or ""
            if not t:
                continue
            pairs.append((t, tag))
    return pairs


def main():
    ensure_nltk()
    lemmatizer = WordNetLemmatizer()
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set(["the","a","an","in","on","at","for","to","and","or"])

    intents = load_intents(DATASET)
    pairs = build_dataset(intents)

    rows = []
    for text, tag in pairs:
        cleaned = clean_text(text)
        if not cleaned or not tag:
            continue
        processed = tokenize_lemmatize(cleaned, lemmatizer, stop_words)
        if not processed:
            continue
        rows.append({"text": processed, "label": tag})

    if not rows:
        raise SystemExit("No data extracted from intents.json")

    df = pd.DataFrame(rows)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "training_data.csv", index=False)
    test_df.to_csv(OUT_DIR / "testing_data.csv", index=False)

    # fit vectorizer on training text only
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    vectorizer.fit(train_df["text"].values)

    le = LabelEncoder()
    le.fit(train_df["label"].values)

    with open(OUT_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(OUT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Preprocessing complete. Saved training_data.csv, testing_data.csv, vectorizer.pkl, label_encoder.pkl")


if __name__ == "__main__":
    main()
