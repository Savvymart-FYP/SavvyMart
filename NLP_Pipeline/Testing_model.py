"""
Fake Review Predictor — With Language Detection & Translation
==============================================================
Supports English, Urdu (script), and Roman Urdu reviews.
Run AFTER model_training.py has been executed.

Install requirements:
    pip install deep-translator langdetect nltk
    python -m nltk.downloader stopwords punkt
"""

import os
import re
import joblib
import numpy as np
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

# Language & Translation
from langdetect import detect
from deep_translator import GoogleTranslator

# Text Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data silently
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

ENGLISH_STOPWORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────
# LOAD FROM SAME DIRECTORY AS THIS SCRIPT
# ─────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = r'E:\SavvyMart\NLP_Pipeline'

MODEL_PATH      = os.path.join(BASE_DIR, "tf_idf_model_1.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH    = os.path.join(BASE_DIR, "label_encoder.pkl")

THRESHOLD = 60

# ─────────────────────────────────────────────
# CHECK & LOAD FILES
# ─────────────────────────────────────────────
for path, name in [
    (MODEL_PATH,      "tf_idf_model_1.pkl"),
    (VECTORIZER_PATH, "tfidf_vectorizer.pkl"),
    (ENCODER_PATH,    "label_encoder.pkl"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  ERROR: {name} not found.\n  Please run model_training.py first.\n"
        )

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)
le    = joblib.load(ENCODER_PATH)

# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────
def detect_language(text: str) -> str:
    # Check for Urdu Unicode script characters
    urdu_chars = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
    if urdu_chars.search(text):
        return "Urdu"
    try:
        code = detect(text)
        if code == "en":
            return "English"
        else:
            return "Roman Urdu"
    except:
        return "Roman Urdu"

# ─────────────────────────────────────────────
# TRANSLATION
# ─────────────────────────────────────────────
def translate_to_english(text: str, lang: str) -> str:
    if lang == "English":
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated if translated else text
    except:
        return text

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove emojis
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
    return " ".join(tokens)

# ─────────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────────
def predict_review(review_text: str, rating: int) -> None:

    # Step 1: Detect language
    lang = detect_language(review_text)

    # Step 2: Translate to English
    translated = translate_to_english(review_text, lang)

    # Step 3: Preprocess (backend only)
    cleaned = preprocess(translated)

    # Step 4: Vectorize & predict
    vec    = tfidf.transform([cleaned])
    extras = np.array([[rating, len(cleaned), len(cleaned.split())]])
    X_new  = sp.hstack([vec, sp.csr_matrix(extras)])

    proba     = model.predict_proba(X_new)[0]
    fake_conf = proba[0] * 100
    real_conf = proba[1] * 100

    # Step 5: Verdict
    if real_conf >= THRESHOLD:
        verdict_tag = "[REAL]"
        verdict     = "REAL  -  Human Written (OR)"
    elif fake_conf >= THRESHOLD:
        verdict_tag = "[FAKE]"
        verdict     = "FAKE  -  Computer Generated (CG)"
    else:
        verdict_tag = "[SUSPICIOUS]"
        verdict     = "SUSPICIOUS  -  Cannot be determined confidently"

    BAR      = 40
    fake_bar = "X" * int(fake_conf / 100 * BAR)
    real_bar = "X" * int(real_conf / 100 * BAR)

    # ─────────────────────────────────────────
    # OUTPUT — only what matters
    # ─────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  PREDICTION RESULT")
    print("=" * 58)
    print(f"  Detected Language  : {lang}")
    print(f"  Translated Review  : \"{translated[:65]}{'...' if len(translated) > 65 else ''}\"")
    print(f"  Rating             : {rating} / 5")
    print(f"\n  {verdict_tag}  {verdict}")
    print(f"  Confidence         : {max(fake_conf, real_conf):.2f}%")
    print(f"\n  Fake (CG)  [{fake_bar.ljust(BAR)}]  {fake_conf:.2f}%")
    print(f"  Real (OR)  [{real_bar.ljust(BAR)}]  {real_conf:.2f}%")
    print("=" * 58)

# ─────────────────────────────────────────────
# USER INPUT LOOP
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("       FAKE REVIEW PREDICTOR")
print("=" * 58)
print("  Supports: English | Urdu | Roman Urdu")
print("  Type 'quit' to exit.")
print("=" * 58)

while True:
    print("\n" + "-" * 58)
    review = input("  Enter Review       : ").strip()

    if review.lower() in ("quit", "exit", "q"):
        print("\n  Goodbye!\n")
        break

    if not review:
        print("  No review entered. Please try again.")
        continue

    while True:
        rating_input = input("  Enter Rating (1-5) : ").strip()
        if rating_input.isdigit() and int(rating_input) in range(1, 6):
            rating = int(rating_input)
            break
        print("  Please enter a number between 1 and 5.")

    predict_review(review, rating)

    print("\n  Test another review? (yes / no) : ", end="")
    if input().strip().lower() in ("no", "n"):
        print("\n  Goodbye!\n")
        break