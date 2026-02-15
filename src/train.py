from __future__ import annotations
import argparse
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.utils import clean_text

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None))
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV with columns: text,label")
    ap.add_argument("--model-out", required=True, help="Output path for trained model (.joblib)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: text,label")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))

    dump(pipe, args.model_out)
    print(f"Saved model -> {args.model_out}")

if __name__ == "__main__":
    main()
