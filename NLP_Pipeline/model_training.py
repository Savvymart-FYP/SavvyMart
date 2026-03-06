"""
Fake Review Detector — TF-IDF + XGBoost Pipeline
==================================================
Predicts whether a review is Original (OR) or Computer-Generated/Fake (CG).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import scipy.sparse as sp

# ─────────────────────────────────────────────
# SAVE DIRECTORY
# ─────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = r'E:\SavvyMart\NLP_Pipeline'

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

FILE_PATH = r'E:\SavvyMart\NLP_Pipeline\fake_reviews_dataset.csv'

try:
    if FILE_PATH.endswith(".tsv"):
        df = pd.read_csv(FILE_PATH, sep="\t")
    elif FILE_PATH.endswith(".xlsx"):
        df = pd.read_excel(FILE_PATH)
    else:
        df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"File '{FILE_PATH}' not found. Update FILE_PATH at the top of the script."
    )

print(f"Dataset shape: {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(df.head(3))

# ─────────────────────────────────────────────
# STEP 2: EDA / SANITY CHECKS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

print("\nNull values:\n",           df.isnull().sum())
print("\nLabel distribution:\n",    df["label"].value_counts())
print("\nCategory distribution:\n", df["category"].value_counts())
print("\nRating distribution:\n",   df["rating"].value_counts().sort_index())

# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

df = df.dropna(subset=["text_", "label"])
df["text_"]       = df["text_"].astype(str)
df["text_length"] = df["text_"].apply(len)
df["word_count"]  = df["text_"].apply(lambda x: len(x.split()))

print(f"\nAvg text length by label:\n{df.groupby('label')['text_length'].mean()}")
print(f"\nAvg word count  by label:\n{df.groupby('label')['word_count'].mean()}")

# ─────────────────────────────────────────────
# STEP 4: TF-IDF ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: TF-IDF Vectorization Analysis")
print("=" * 60)

_probe = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
_probe.fit(df["text_"])

vocab_size = len(_probe.vocabulary_)
print(f"Vocabulary size (min_df=2, max_df=0.95, 1-2 grams): {vocab_size:,}")

_mat         = _probe.transform(df["text_"])
scores       = np.asarray(_mat.mean(axis=0)).flatten()
token_scores = sorted(
    zip(_probe.get_feature_names_out(), scores),
    key=lambda x: x[1], reverse=True
)[:20]
print("\nTop-20 tokens by mean TF-IDF score:")
for tok, sc in token_scores:
    print(f"  {tok:<30} {sc:.5f}")

TFIDF_PARAMS = dict(
    ngram_range  = (1, 2),
    min_df       = 2,
    max_df       = 0.95,
    max_features = 10_000,
    sublinear_tf = True
)
print(f"\nFinal TF-IDF params: {TFIDF_PARAMS}")

# ─────────────────────────────────────────────
# STEP 5: ENCODE LABELS & NUMERIC FEATURES
# ─────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(df["label"])
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

extra = df[["rating", "text_length", "word_count"]].values

# ─────────────────────────────────────────────
# STEP 6: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Train/Test Split (80/20, stratified)")
print("=" * 60)

texts_train, texts_test, extra_train, extra_test, y_train, y_test = \
    train_test_split(
        df["text_"], extra, y,
        test_size=0.2, random_state=42, stratify=y
    )

print(f"Train size : {len(texts_train):,}")
print(f"Test  size : {len(texts_test):,}")

# ─────────────────────────────────────────────
# STEP 7: FIT TF-IDF & BUILD FEATURE MATRIX
# ─────────────────────────────────────────────
tfidf         = TfidfVectorizer(**TFIDF_PARAMS)
X_train_tfidf = tfidf.fit_transform(texts_train)
X_test_tfidf  = tfidf.transform(texts_test)

X_train = sp.hstack([X_train_tfidf, sp.csr_matrix(extra_train)])
X_test  = sp.hstack([X_test_tfidf,  sp.csr_matrix(extra_test)])

print(f"\nFeature matrix shape - train: {X_train.shape}  |  test: {X_test.shape}")

# ─────────────────────────────────────────────
# STEP 8: TRAIN XGBOOST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Training XGBoost Model")
print("=" * 60)

neg, pos = np.bincount(y_train)
scale_pw  = neg / pos
print(f"Class balance - CG: {neg}, OR: {pos}  |  scale_pos_weight: {scale_pw:.2f}")

model = XGBClassifier(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_pw,
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = 42,
    n_jobs            = -1,
    tree_method       = "hist"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

print("\nTraining complete.")

# ─────────────────────────────────────────────
# STEP 9: EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Model Evaluation")
print("=" * 60)

def evaluate(X, y_true, split_name):
    y_pred = model.predict(X)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec  = recall_score(y_true, y_pred, average="weighted")
    f1   = f1_score(y_true, y_pred, average="weighted")
    print(f"\n{'─'*40}")
    print(f"{split_name} Results")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return y_pred

y_train_pred = evaluate(X_train, y_train, "TRAINING")
y_test_pred  = evaluate(X_test,  y_test,  "TESTING")

# ─────────────────────────────────────────────
# STEP 10: CONFUSION MATRICES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Confusion Matrix")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices - Fake Review Detector", fontsize=14, fontweight="bold")

for ax, y_true, y_pred, title in [
    (axes[0], y_train, y_train_pred, "Training Set"),
    (axes[1], y_test,  y_test_pred,  "Test Set"),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
cm_path = os.path.join(BASE_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved -> {cm_path}")
plt.show()

# ─────────────────────────────────────────────
# STEP 11: PREDICT ON NEW REVIEWS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Predict on New Reviews")
print("=" * 60)

def predict_review(review_text: str, rating: int = 5) -> str:
    """Pass a raw review string and get CG / OR prediction."""
    vec    = tfidf.transform([review_text])
    extras = np.array([[rating, len(review_text), len(review_text.split())]])
    X_new  = sp.hstack([vec, sp.csr_matrix(extras)])
    pred   = model.predict(X_new)[0]
    prob   = model.predict_proba(X_new)[0]
    label  = le.inverse_transform([pred])[0]
    conf   = prob[pred] * 100
    return f"Prediction: {label}  (confidence: {conf:.1f}%)"

samples = [
    ("love this product absolutely perfect quality great buy", 5),
    ("item arrived broken poor quality waste of money", 1),
    ("great product love look feel pillow nice size", 5),
]
for text, rating in samples:
    print(f"\nReview : '{text[:60]}...'")
    print(predict_review(text, rating))

# ─────────────────────────────────────────────
# STEP 12: SAVE MODEL, VECTORIZER & LABEL ENCODER
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Saving Model & Vectorizer")
print("=" * 60)

joblib.dump(model, os.path.join(BASE_DIR, "tf_idf_model_1.pkl"))
joblib.dump(tfidf, os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(le,    os.path.join(BASE_DIR, "label_encoder.pkl"))

print(f"  model saved         -> {os.path.join(BASE_DIR, 'tf_idf_model_1.pkl')}")
print(f"  vectorizer saved    -> {os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')}")
print(f"  label encoder saved -> {os.path.join(BASE_DIR, 'label_encoder.pkl')}")
print("\n  Run predict_review.py to test the model")