# src/train_and_eval.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH  = "data/processed/test.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg.joblib")
FIG_DIR    = "models"   # where to save confusion matrix image

def load_xy(path):
    df = pd.read_csv(path)
    X = df["text"].astype(str).tolist()
    y = df["label"].tolist()
    return X, y

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train = load_xy(TRAIN_PATH)
    X_test,  y_test  = load_xy(TEST_PATH)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_df=0.9,
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=500))
    ])

    print("Training...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print(f"\nAccuracy: {acc:.3f}")
    print(f"Precision (weighted): {prec:.3f}")
    print(f"Recall (weighted): {rec:.3f}")
    print(f"F1 (weighted): {f1:.3f}\n")

    print("Per-class report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion matrix figure
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE","REAL"])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAKE","REAL"], yticklabels=["FAKE","REAL"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - TFIDF + LogisticRegression")
    fig_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print("Saved confusion matrix to:", fig_path)

    # Save model pipeline (includes the TF-IDF vectorizer)
    joblib.dump(pipe, MODEL_PATH)
    print("Saved model to:", MODEL_PATH)

if __name__ == "__main__":
    main()
