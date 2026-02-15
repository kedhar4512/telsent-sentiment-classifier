from __future__ import annotations
import argparse
from joblib import load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    model = load(args.model)
    pred = model.predict([args.text])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([args.text])[0].max()

    if proba is None:
        print(f"Prediction: {pred}")
    else:
        print(f"Prediction: {pred} (confidence ~ {proba:.2f})")

if __name__ == "__main__":
    main()
