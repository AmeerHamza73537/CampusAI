import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


ROOT = Path(".")
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    # cross-val on training
    cv_scores = None
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    except Exception:
        cv_scores = []

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    return {
        "name": name,
        "cv_scores": cv_scores.tolist() if hasattr(cv_scores, "tolist") else list(cv_scores),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "model": model,
    }


def build_and_train_nn(X_train, y_train, X_test, y_test, num_classes):
    # convert to dense arrays
    X_train_a = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_a = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(X_train_a.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 

    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    model.fit(X_train_a, y_train, validation_split=0.1, epochs=50, batch_size=16, callbacks=[es], verbose=1)

    y_pred_train = np.argmax(model.predict(X_train_a), axis=1)
    y_pred_test = np.argmax(model.predict(X_test_a), axis=1)

    train_acc = float((y_pred_train == y_train).mean())
    test_acc = float((y_pred_test == y_test).mean())
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    return {
        "name": "NeuralNet",
        "train_acc": train_acc,
        "test_acc": test_acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "model": model,
    }


def main():
    # load data
    train_df = pd.read_csv(ROOT / "training_data.csv")
    test_df = pd.read_csv(ROOT / "testing_data.csv")

    with open(ROOT / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(ROOT / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    X_train = vectorizer.transform(train_df["text"].values)
    X_test = vectorizer.transform(test_df["text"].values)
    y_train = le.transform(train_df["label"].values)
    y_test = le.transform(test_df["label"].values)

    results = []

    # Multinomial Naive Bayes
    nb = MultinomialNB()
    r_nb = evaluate_model("MultinomialNB", nb, X_train, y_train, X_test, y_test)
    results.append(r_nb)
    with open(RESULTS / "model_nb.pkl", "wb") as f:
        pickle.dump(r_nb["model"], f)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    r_rf = evaluate_model("RandomForest", rf, X_train, y_train, X_test, y_test)
    results.append(r_rf)
    with open(RESULTS / "model_rf.pkl", "wb") as f:
        pickle.dump(r_rf["model"], f)

    # Neural Network (only if TensorFlow available)
    if TF_AVAILABLE:
        nn_res = build_and_train_nn(X_train, y_train, X_test, y_test, num_classes=len(le.classes_))
        results.append(nn_res)
        # save keras model
        nn_res["model"].save(RESULTS / "model_nn.h5")
    else:
        print("TensorFlow not available; skipping neural network training.")

    # pick best by test_acc
    best = max(results, key=lambda r: r["test_acc"]) 
    # save best scikit model or note keras
    if best["name"] == "NeuralNet":
        with open(RESULTS / "best_model_info.json", "w") as f:
            json.dump({"best": "NeuralNet"}, f)
    else:
        # save the selected pickle in results/best_model.pkl
        with open(RESULTS / "best_model.pkl", "wb") as f:
            pickle.dump(best["model"], f)
        with open(RESULTS / "best_model_info.json", "w") as f:
            json.dump({"best": best["name"]}, f)

    # save metrics and confusion matrices
    metrics = {r["name"]: {k: r[k] for k in ["train_acc","test_acc","precision","recall","f1","confusion_matrix"]} for r in results}
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # save confusion matrices as npy
    for r in results:
        cm = np.array(r["confusion_matrix"])
        np.save(RESULTS / f"confusion_{r['name']}.npy", cm)

    print("Training complete. Results saved to results/")


if __name__ == "__main__":
    main()
