
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Project folders
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "spam.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# 1. Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()


# 2. Load dataset
def load_data():

    print("Loading dataset...", flush=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError("data/spam.csv not found!")

    df = pd.read_csv(DATA_PATH)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain text and label columns.")

    df = df[["text", "label"]].dropna().copy()

    df["text"] = df["text"].astype(str).str.lower()
    df["text"] = df["text"].str.replace(
        r"http\S+|www\.\S+", " URL ", regex=True
    )
    df["text"] = df["text"].str.replace(
        r"\S+@\S+", " EMAIL ", regex=True
    )
    df["text"] = df["text"].str.replace(
        r"[^a-z0-9\s]", " ", regex=True
    )

    df["label"] = df["label"].astype(str).str.lower().str.strip()

    df = df[df["label"].isin(["spam", "ham"])]
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"])

    print("Total emails:", len(df), flush=True)
    print("\nEmail distribution:")
    print(df["label"].value_counts(), flush=True)

    return df


# 3. Train and evaluate models
def main():

    df = load_data()

    print("\nSplitting dataset...", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    print("Training emails:", len(X_train))
    print("Testing emails:", len(X_test))

    # Shared TF-IDF settings
    models = {
        "Naive Bayes": MultinomialNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    results = []

    # 4. Model training
    for name, classifier in models.items():

        print(f"\nTraining {name}...", flush=True)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                min_df=2,
                ngram_range=(1, 1),
                sublinear_tf=True,
                dtype=float
            )),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        print(f"Testing {name}...", flush=True)

        predictions = pipeline.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions,
            pos_label="spam",
            zero_division=0
        )

        recall = recall_score(
            y_test, predictions,
            pos_label="spam",
            zero_division=0
        )

        f1 = f1_score(
            y_test, predictions,
            pos_label="spam",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

        print(f"\n{name} Results:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

        # Save trained model
        filename = name.lower().replace(" ", "_") + ".joblib"

        joblib.dump(
            pipeline,
            MODEL_DIR / filename
        )

        # Confusion matrix
        cm = confusion_matrix(
            y_test,
            predictions,
            labels=["ham", "spam"]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Ham", "Spam"]
        )

        display.plot(values_format="d", cmap="Blues")

        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()

        plt.savefig(
            RESULT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png",
            dpi=150
        )

        plt.close()

        print(f"{name} completed successfully!", flush=True)

    # 5. Save comparison results
    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="F1-score",
        ascending=False
    )

    comparison.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    # 6. Generate comparison graph
    ax = comparison.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1-score"]
    ].plot(
        kind="bar",
        figsize=(11, 6),
        ylim=(0, 1)
    )

    ax.set_title("Machine Learning Model Performance Comparison")
    ax.set_ylabel("Performance Score")
    ax.set_xlabel("Machine Learning Algorithms")
    ax.legend(loc="lower right")

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "metrics_comparison.png",
        dpi=150
    )

    plt.close()

    print("\n============================================")
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("Models saved in:", MODEL_DIR)
    print("Results saved in:", RESULT_DIR)
    print("============================================")


if __name__ == "__main__":
    main()
