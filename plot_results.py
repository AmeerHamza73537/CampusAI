import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(".")
RESULTS = ROOT / "results"


def plot_metrics(metrics_path: Path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    names = list(metrics.keys())
    train_acc = [metrics[n]["train_acc"] for n in names]
    test_acc = [metrics[n]["test_acc"] for n in names]
    precision = [metrics[n]["precision"] for n in names]
    recall = [metrics[n]["recall"] for n in names]
    f1 = [metrics[n]["f1"] for n in names]

    x = range(len(names))
    plt.figure(figsize=(8,5))
    plt.bar(x, train_acc, width=0.35, label="train_acc")
    plt.bar([i+0.35 for i in x], test_acc, width=0.35, label="test_acc")
    plt.xticks([i+0.17 for i in x], names)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison_accuracy.png")
    plt.close()

    # other metrics
    plt.figure(figsize=(8,5))
    width = 0.2
    plt.bar([i-width for i in x], precision, width=width, label="precision")
    plt.bar(x, recall, width=width, label="recall")
    plt.bar([i+width for i in x], f1, width=width, label="f1")
    plt.xticks(x, names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison_prf.png")
    plt.close()


def plot_confusion(cm_path: Path, name: str):
    cm = np.load(cm_path)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(RESULTS / f"confusion_{name}.png")
    plt.close()


def main():
    metrics_path = RESULTS / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("metrics.json not found; run train_models.py first")
    plot_metrics(metrics_path)

    for p in RESULTS.glob("confusion_*.npy"):
        name = p.stem.replace("confusion_", "")
        plot_confusion(p, name)

    print("Plots saved to results/")


if __name__ == "__main__":
    main()
