import os
import re
import joblib
import numpy as np
import scipy.sparse as sp
from groq import Groq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import nltk

# Initialize dependencies
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))
load_dotenv()

# 1. Load Models & API
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    vectorizer = joblib.load(os.path.join(current_dir, 'tfidf_vectorizer.pkl'))
    model = joblib.load(os.path.join(current_dir, 'tf_idf_model_1.pkl'))
    encoder = joblib.load(os.path.join(current_dir, 'label_encoder.pkl'))
    client = Groq(api_key=os.getenv("MY_SECRET_API_KEY"))
    print("System ready for live testing.")
except Exception as e:
    print(f"Error loading assets: {e}")
    exit()

# 2. Core Functions
def translate_to_english(text):
    prompt = "Translate Roman Urdu/slang to English. Return ONLY the translation."
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except:
        return None

def preprocess_text(text):
    text = text.lower().encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [t for t in word_tokenize(text) if t not in stop_words]
    return " ".join(tokens)

# 3. Unit Test (Automated check)
def run_unit_test():
    sample = "BKWS! product."
    result = preprocess_text(sample)
    if "!" not in result and result == result.lower():
        print("Unit Test: Preprocessing OK")
    else:
        raise ValueError("Preprocessing logic failed.")

run_unit_test()

# 4. Live Testing Loop
print("-" * 30)
while True:
    user_review = input("\nEnter Roman Urdu Review (or 'exit'): ").strip()
    if user_review.lower() == 'exit': break
    
    try:
        rating = float(input("Enter Rating (1-5): "))
        
        # Pipeline: Translation -> Preprocessing -> Feature Extraction
        translated = translate_to_english(user_review)
        if not translated: 
            print("Translation Error.")
            continue
            
        cleaned = preprocess_text(translated)
        
        # Create final feature vector
        tfidf_feat = vectorizer.transform([cleaned])
        extras = np.array([[rating, len(cleaned), len(cleaned.split())]])
        final_input = sp.hstack([tfidf_feat, sp.csr_matrix(extras)])
        
        # Prediction
        pred = model.predict(final_input)
        label = encoder.inverse_transform(pred)
        
        print(f"English: {translated}")
        print(f"Prediction: {label[0].upper()}")
    except Exception as e:
        print(f"Pipeline Error: {e}")