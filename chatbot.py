import json
import random
import pickle
from pathlib import Path

import numpy as np
from preprocess_data import clean_text, tokenize_lemmatize, ensure_nltk

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ROOT = Path(".")
RESULTS = ROOT / "results"


class ChatbotEngine:
    def __init__(self, model_path=None):
        ensure_nltk()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        with open(ROOT / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(ROOT / "label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)

        # load best model if available
        # prefer results/best_model.pkl or results/model_nn.h5
        best_pickle = RESULTS / "best_model.pkl"
        best_nn = RESULTS / "model_nn.h5"
        if best_pickle.exists():
            with open(best_pickle, "rb") as f:
                self.model = pickle.load(f)
            self.model_type = "sklearn"
        elif best_nn.exists():
            from tensorflow import keras
            self.model = keras.models.load_model(str(best_nn))
            self.model_type = "keras"
        else:
            # fallback: try individual models
            self.model = None
            self.model_type = None

        # load intents for responses
        with open(ROOT / "dataset" / "intents.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # build mapping tag -> responses
        intents = data.get("intents") if isinstance(data, dict) else data
        self.responses = {}
        if isinstance(intents, dict):
            intents = intents.get("intents", [])
        for it in intents:
            tag = it.get("tag") or it.get("intent") or it.get("name")
            res = it.get("responses") or []
            self.responses[tag] = res

        self.direct_intents = {
            "hi": "greeting",
            "hello": "greeting",
            "hey": "greeting",
            "hey there": "greeting",
            "hi there": "greeting",
            "good morning": "greeting",
            "good afternoon": "greeting",
            "good evening": "greeting",
            "bye": "goodbye",
            "cya": "goodbye",
            "see ya": "goodbye",
            "see ya later": "goodbye",
            "take care": "goodbye",
            "talk soon": "goodbye",
            "what if i miss an exam": "exam_policy",
            "what if i miss my exam": "exam_policy",
            "miss an exam": "exam_policy",
            "what if i miss my exam due to illness": "exam_policy",
            "how can i check my exam result": "results",
            "how do i check my exam result": "results",
            "exam result": "results",
            "exam results": "results"
        }

    def preprocess(self, text: str) -> str:
        cleaned = clean_text(text)
        return tokenize_lemmatize(cleaned, self.lemmatizer, self.stop_words)

    def predict(self, text: str):
        normalized = " ".join(text.strip().lower().split())
        if normalized in self.direct_intents:
            tag = self.direct_intents[normalized]
            responses = self.responses.get(tag) or ["Sorry, I don't know that yet."]
            return {"intent": tag, "response": random.choice(responses)}

        processed = self.preprocess(text)
        X = self.vectorizer.transform([processed])
        if self.model is None:
            return {"intent": None, "response": "Sorry, model not found."}

        confidence = 1.0
        if self.model_type == "sklearn":
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                confidence = float(np.max(probs))
            pred_idx = self.model.predict(X)[0]
            pred = self.le.inverse_transform([pred_idx])[0]
        else:
            arr = X.toarray()
            probs = self.model.predict(arr)[0]
            confidence = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            pred = self.le.inverse_transform([pred_idx])[0]

        intent = str(pred)
        responses = self.responses.get(pred) or []

        if confidence < 0.30 or not responses:
            fallback = [
                "I'm designed to answer university-related questions. Please ask something about admissions, programs, fees, campus facilities, exams, results, or student services.",
                "Sorry, I didn't understand that. Can you ask something related to university?",
                "I am not sure about that. Please ask another university-related question."
            ]
            return {"intent": "unknown", "response": random.choice(fallback)}

        return {"intent": intent, "response": random.choice(responses)}


if __name__ == "__main__":
    bot = ChatbotEngine()
    while True:
        try:
            text = input("You: ")
        except EOFError:
            break
        out = bot.predict(text)
        print("Bot:", out["response"]) 
