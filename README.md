# TelSent — Sentiment Classifier (Python + ML)

An ML project demonstrating:
- Text preprocessing
- TF‑IDF features
- Logistic Regression classifier
- Evaluation (accuracy, precision/recall, confusion matrix)
- Reproducible training and inference pipeline

## Tech
Python, scikit-learn, pandas

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Train on sample data
python src/train.py --data data/sample_reviews.csv --model-out models/sentiment.joblib

# Evaluate
python src/evaluate.py --data data/sample_reviews.csv --model models/sentiment.joblib

# Predict
python src/predict.py --model models/sentiment.joblib --text "This hotel is awesome, super clean!"
```

## Data format
CSV with columns:
- `text` (string)
- `label` (one of: positive, negative, neutral)

## Notes
Replace `data/sample_reviews.csv` with your dataset (Kaggle/your own). Keep the same schema.
