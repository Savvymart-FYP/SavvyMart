import pandas as pd
import joblib
import re
import os
import numpy as np

# Path setup (taki ye sub-folder se project root ki files dhoond sakay)
MODELS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(MODELS_DIR)

# --- Global Variables ---
model = None
tfidf = None
model_loaded = False

# --- Load Models Once ---
try:
    model = joblib.load(os.path.join(PROJECT_ROOT, 'sentiment_model.pkl'))
    tfidf = joblib.load(os.path.join(PROJECT_ROOT, 'tfidf_vectorizer.pkl'))
    model_loaded = True
    print("Sentiment Models Loaded Successfully into Module!")
except Exception as e:
    print(f"Module Model Load Error: {e}")

def clean_text(text):
    if not text or str(text).lower() == 'nan' or str(text).lower() == 'none':
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    corrections = {
        # --- BASICS & CONNECTORS ---
        'h': 'hai', 'hy': 'hai', 'ha': 'hai', 'hen': 'hain', 'hn': 'hain', 'hai': 'hai', 'hain': 'hain',
        'ho': 'ho', 'hu': 'hoon', 'hoon': 'hoon', 'tha': 'tha', 'thi': 'thi', 'thy': 'the', 'the': 'the',
        'k': 'ke', 'ka': 'ka', 'ki': 'ki', 'ke': 'ke', 'ko': 'ko', 'kee': 'ki',
        'b': 'bhi', 'bh': 'bhi', 'bi': 'bhi', 'bhi': 'bhi',
        'or': 'aur', 'r': 'aur', 'aur': 'aur', 'nd': 'aur', 'and': 'aur',
        'pr': 'par', 'par': 'par', 'pe': 'par',
        'sy': 'se', 'se': 'se', 'st': 'se', 'se': 'se',
        'sth': 'sath', 'sath': 'sath', 'saath': 'sath',
        'tk': 'tak', 'tak': 'tak',
        'lkn': 'lekin', 'likin': 'lekin', 'lekin': 'lekin',

        # --- NEGATIONS ---
        'nhi': 'nahi', 'ni': 'nahi', 'nh': 'nahi', 'nai': 'nahi', 'nehi': 'nahi', 'na': 'na',

        # --- PRONOUNS (All Variations) ---
        'm': 'mein', 'ma': 'mein', 'mai': 'mein', 'me': 'mein', 'mein': 'mein',
        'ap': 'aap', 'apki': 'aapki', 'apka': 'aapka', 'apke': 'aapke', 'aap': 'aap',
        'tm': 'tum', 'tmhara': 'tumhara', 'tumhara': 'tumhara',
        'tujhe': 'tujhe', 'tjy': 'tujhe', 'tujhy': 'tujhe', 'tjhe': 'tujhe',
        'mjy': 'mujhe', 'mjhe': 'mujhe', 'mujhy': 'mujhe', 'mj': 'mujhe', 'muj': 'mujhe',
        'hm': 'hum', 'hum': 'hum', 'hmary': 'hamare', 'hmra': 'humara',
        'wo': 'wo', 'vo': 'wo', 'v': 'wo', 'un': 'un', 'inhu': 'inho', 'unhu': 'unho',

        # --- ACTION VERBS & REVIEWS SPECIFIC ---
        'kr': 'kar', 'karo': 'karo', 'kro': 'karo', 'krna': 'karna', 'karna': 'karna', 'krty': 'karte',
        'karty': 'karte',
        'rha': 'raha', 'rhi': 'rahi', 'rhe': 'rahe', 'raha': 'raha',
        'ata': 'aata', 'ati': 'aati', 'aty': 'aate',
        'pta': 'pata', 'pata': 'pata',
        'gya': 'gaya', 'gyi': 'gayi', 'gye': 'gaye', 'gaya': 'gaya',
        'lia': 'liya', 'ly': 'liya', 'liya': 'liya', 'lya': 'liya',
        'dia': 'diya', 'dy': 'diya', 'diya': 'diya',
        'le': 'le', 'lo': 'lo', 'de': 'de', 'do': 'do',
        'chaye': 'chahiye', 'chaiye': 'chahiye', 'chahiye': 'chahiye', 'chahye': 'chahiye',
        'mngwaya': 'mangwaya', 'mangwaya': 'mangwaya', 'ordr': 'order', 'mil': 'mila', 'mila': 'mila',
        'shawer': 'shower', 'shower': 'shower', 'shaur': 'shower',  # Added for your review fix

        # --- POSITIVE SENTIMENTS & QUALITY (Everything included) ---
        'bht': 'bohat', 'boht': 'bohat', 'bhut': 'bohat', 'bt': 'bohat', 'bot': 'bohat', 'bohat': 'bohat',
        'zbrdst': 'zabardast', 'zabrdst': 'zabardast', 'zbardast': 'zabardast', 'zb': 'zabardast',
        'zabardast': 'zabardast',
        'thk': 'theek', 'thek': 'theek', 'tik': 'theek', 'theek': 'theek',
        'acha': 'acha', 'achha': 'acha', 'achaa': 'acha', 'acha': 'acha',
        'psnd': 'pasand', 'pasand': 'pasand', 'psand': 'pasand',
        'gud': 'good', 'gd': 'good', 'good': 'good',
        'awsm': 'awesome', 'osm': 'awesome', 'awesome': 'awesome',
        'nyc': 'nice', 'nice': 'nice', 'fit': 'fit', 'kmaal': 'kamal', 'kamal': 'kamal',
        'umda': 'umda', 'umdah': 'umda', 'shandaar': 'shandar',
        'maze': 'mazay', 'mazay': 'mazay', 'maazay': 'mazay', 'mazaydar': 'mazay',  # Fix for "maze k"
        'asli': 'asli', 'original': 'original', 'orignal': 'original', 'gnun': 'genuine',
        'promising': 'acha', 'texture': 'acha', 'quantity': 'zyada',  # Fix for product reviews
        'ingrediant': 'ingredient', 'ingredient': 'ingredient',  # Fix for typo in review

        # --- NEGATIVE SENTIMENTS ---
        'bkws': 'bakwas', 'bakwass': 'bakwas', 'bakwas': 'bakwas',
        'ghtya': 'ghatiya', 'ghateya': 'ghatiya', 'ghtiya': 'ghatiya', 'ghatya': 'ghatiya', 'ghatiya': 'ghatiya',
        'fzul': 'fazool', 'fzol': 'fazool', 'fazul': 'fazool', 'fazool': 'fazool',
        'khrb': 'kharab', 'khrab': 'kharab', 'khraab': 'kharab', 'kharab': 'kharab',
        'gnda': 'ganda', 'gndy': 'ganday', 'gandy': 'ganday', 'ganda': 'ganda',
        'froud': 'fraud', 'fraud': 'fraud', 'dhoka': 'dhoka', 'fake': 'fake',
        'bekaar': 'bekaar', 'bekar': 'bekaar', 'ghalat': 'galat', 'galat': 'galat',
        'jhoot': 'jhoot', 'expire': 'expiry', 'expry': 'expiry',

        # --- TIME, SOCIAL & MISC ---
        'yr': 'yaar', 'yar': 'yaar', 'yaar': 'yaar',
        'ab': 'ab', 'avi': 'abhi', 'abhi': 'abhi', 'abhe': 'abhi',
        'kl': 'kal', 'aj': 'aaj', 'aaj': 'aaj',
        'idr': 'idhar', 'idhr': 'idhar', 'idhar': 'idhar',
        'udr': 'udhar', 'udhr': 'udhar', 'udhar': 'udhar',
        'kdr': 'kidhar', 'kdhr': 'kidhar', 'kidhar': 'kidhar',
        'kb': 'kab', 'jb': 'jab',
        'aik': 'ek', 'ek': 'ek',
        'q': 'kyun', 'kyu': 'kyun', 'kyun': 'kyun', 'qn': 'kyun',
        'wla': 'wala', 'wli': 'wali', 'wle': 'wale', 'wala': 'wala',
        'thx': 'thanks', 'thnx': 'thanks', 'ty': 'thanks', 'thanks': 'thanks',
        'shukria': 'shukriya', 'shukrya': 'shukriya', 'shukriya': 'shukriya',
        'plz': 'please', 'pls': 'please', 'please': 'please',
        'srf': 'sirf', 'sirf': 'sirf',
        'esa': 'aisa', 'aysa': 'aisa', 'aisa': 'aisa',
        'wse': 'waise', 'wayse': 'waise', 'waise': 'waise',
        'shd': 'shayad', 'shyd': 'shayad', 'shayad': 'shayad'
    }

    words = text.split()
    fixed_words = [corrections.get(w, w) for w in words]
    return " ".join(fixed_words).strip()

def get_sentiment_percentages(reviews_list):
    """
    Har product ke reviews ki list leta hai aur [Pos, Neg, Neu] percentages return karta hai.
    """
    if not model_loaded or not reviews_list:
        return pd.Series([0, 0, 0])

    pos, neg, neu = 0, 0, 0
    valid_reviews = 0

    for rev in reviews_list:
        body = rev.get('Review Body')
        if body:
            cleaned = clean_text(body)
            if cleaned:
                vectorized = tfidf.transform([cleaned])
                sentiment = model.predict(vectorized)[0].lower()

                if sentiment == 'positive': pos += 1
                elif sentiment == 'negative': neg += 1
                else: neu += 1
                valid_reviews += 1

    if valid_reviews == 0:
        return pd.Series([0, 0, 0])

    pos_pct = round((pos / valid_reviews) * 100)
    neg_pct = round((neg / valid_reviews) * 100)
    neu_pct = round((neu / valid_reviews) * 100)

    return pd.Series([pos_pct, neg_pct, neu_pct])