from pathlib import Path
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "spam.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def clean_text(text):
    """Basic text cleanup; TF-IDF handles tokenization and weighting."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Add data/spam.csv with columns named text and label."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns named 'text' and 'label'.")
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["spam", "ham"])]
    df = df[df["text"].str.len() > 0]
    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both 'spam' and 'ham' labels.")
    if len(df) < 20:
        print("WARNING: Very small dataset. Use a larger dataset for research.")
    return df


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    rows = []
    for name, classifier in classifiers.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english", max_features=20000,
                ngram_range=(1, 2), sublinear_tf=True
            )),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        scores = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
            "Recall": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
            "F1-score": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        }
        rows.append(scores)
        joblib.dump(pipeline, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

        cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
        display = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

        print(f"\n{name}\n{classification_report(y_test, predictions, zero_division=0)}")

    results = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
    results.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON")
    print(results.to_string(index=False))

    ax = results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", ylim=(0, 1), figsize=(10, 6)
    )
    ax.set_ylabel("Score")
    ax.set_title("Spam Email Classifier Performance Comparison")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "metrics_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved models to: {MODEL_DIR}")
    print(f"Saved results to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
