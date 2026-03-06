import os
import re
import string
import joblib
import numpy as np
import scipy.sparse as sp
from groq import Groq
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Download NLTK data automatically
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

# ==========================================
# 0. LOAD MACHINE LEARNING MODELS
# ==========================================
print("Loading Machine Learning Models...")
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    vectorizer = joblib.load(os.path.join(current_dir, 'tfidf_vectorizer.pkl'))
    model = joblib.load(os.path.join(current_dir, 'tf_idf_model_1.pkl'))
    encoder = joblib.load(os.path.join(current_dir, 'label_encoder.pkl'))
    print("Models Loaded Successfully!\n")
except Exception as e:
    print(f"Error loading PKL files: {e}")
    exit()

# ==========================================
# 1. API SETUP
# ==========================================
# Yahan apni asli Groq API key daalein
load_dotenv()
GROQ_API_KEY = os.getenv("MY_SECRET_API_KEY")

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"API Client setup mein masla: {e}")
    exit()

# ==========================================
# 2. CORE TRANSLATION FUNCTION
# ==========================================
def translate_to_english(roman_urdu_text):
    system_prompt = """
    You are an expert linguist and translator. Your ONLY job is to translate Roman Urdu/Hindi text into natural, grammatically correct English.
    
    IMPORTANT RULES:
    1. The input will contain heavy slang, SMS language (e.g., 'rye', 'mjy', 'kia', 'hogea', 'bkws', 'yr'), and typos. Understand the contextual meaning.
    2. Provide ONLY the final English translation.
    3. DO NOT add any conversational text, explanations, or quotes.
    4. If the text is already in English, just return it as is or correct its grammar.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": roman_urdu_text}
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.2, 
            max_tokens=200   
        )
        result = chat_completion.choices[0].message.content.strip(' "')
        return result
    except Exception as e:
        return f"Error: API ya Internet ka masla hai -> {str(e)}"

# ==========================================
# 3. PREPROCESSING FUNCTION
# ==========================================
def preprocess_text(text):
    text = text.lower()
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ==========================================
# 4. LIVE CHAT LOOP & PIPELINE
# ==========================================
print("=" * 65)
print("🚀 SAVVYMART AI FRAUD REVIEW DETECTOR STARTED!")
print("Type 'exit' in the review section to stop the program.")
print("=" * 65)

while True:
    # USER SE SIRF 2 INPUTS: Review aur Rating
    user_input = input("\n📝 Roman Urdu Review: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'stop']:
        print("Khuda Hafiz! Bhai ka program band ho raha hai.")
        break
        
    if not user_input:
        print("Kuch likh toh do bhai!")
        continue
        
    try:
        user_rating = float(input("⭐ Rating (1 to 5): "))
    except ValueError:
        print("⚠️ Invalid rating! Please enter a number.")
        continue
        
    # FLOW START
    print("⏳ Translating (AI dimaagh laga raha hai)...")
    english_translation = translate_to_english(user_input)
    
    if "Error:" in english_translation:
        print(f"Translation Failed: {english_translation}")
        continue
        
    print(f"English    : {english_translation}")

    print("Preprocessing text...")
    cleaned_text = preprocess_text(english_translation)
    print(f"   -> Cleaned : {cleaned_text}")
    
    try:
        print("🧮 Extracting features (Background logic applied)...")
        # 1. TF-IDF features (10000 columns)
        tfidf_features = vectorizer.transform([cleaned_text])
        
        # 2. Background variables calculation (Length and Word Count)
        text_length = len(cleaned_text)
        word_count = len(cleaned_text.split())
        
        # 3. Combining all features: Rating, Length, Word Count (3 columns)
        extras = np.array([[user_rating, text_length, word_count]])
        
        # Total Features: 10000 + 3 = 10003 columns
        final_features = sp.hstack([tfidf_features, sp.csr_matrix(extras)])
        prediction = model.predict(final_features)
        
        # Decoding Prediction (0/1 to Text)
        final_label = encoder.inverse_transform(prediction)
        
        print("-" * 50)
        print(f"This review is {final_label[0].upper()}!")
        print("-" * 50)
        
    except Exception as e:
        print(f"ML Pipeline Error: {e}")