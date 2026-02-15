from __future__ import annotations
import argparse
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    model = load(args.model)
    preds = model.predict(X)

    print("Classification report:")
    print(classification_report(y, preds))

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, preds))

if __name__ == "__main__":
    main()
