import pandas as pd
import os
import re
from flask import Flask, render_template, jsonify, request
import numpy as np
from NLP_Pipeline.sentiment_score import get_sentiment_percentages
import os
from dotenv import load_dotenv



# --- SMART SEARCH IMPORTS ---
from groq import Groq
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

load_dotenv()
GROQ_API_KEY = os.getenv("MY_SECRET_API_KEY")
try:
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Groq Setup Error: {e}")
# ----------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, 'Websites Data', 'data.csv')

def get_processed_data():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        df = pd.read_csv(DATA_FILE)

        review_cols = ['Reviewer Name', 'Rating', 'Review Body', 'Date']
        reviews_dict = df.groupby('Product URL', group_keys=False).apply(
            lambda x: x[review_cols].replace({np.nan: None}).to_dict('records')
        ).to_dict()

        df_unique = df.groupby('Product URL', as_index=False).first()

        df_unique['Rating'] = pd.to_numeric(df_unique['Rating'], errors='coerce').fillna(4.0)
        df_unique['Regular Price'] = pd.to_numeric(df_unique['Regular Price'], errors='coerce').fillna(0)
        df_unique['Sale Price'] = pd.to_numeric(df_unique['Sale Price'], errors='coerce').fillna(0)

        df_unique['All_Reviews'] = df_unique['Product URL'].map(reviews_dict)
        df_unique['Review_Count'] = df_unique['All_Reviews'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        def calc_metrics(row):
            reg, sale = row['Regular Price'], row['Sale Price']
            discount = round(((reg - sale) / reg) * 100) if reg > sale > 0 else 0
            savings = reg - sale if reg > sale else 0
            return pd.Series([discount, savings])

        df_unique[['Discount_Pct', 'Savings']] = df_unique.apply(calc_metrics, axis=1)
        df_unique[['Pos_Pct', 'Neg_Pct', 'Neu_Pct']] = df_unique['All_Reviews'].apply(get_sentiment_percentages)

        return df_unique.replace({np.nan: None}).to_dict('records')
    except Exception as e:
        print(f"Server Error: {e}")
        return []

@app.route('/')
def home():
    data = get_processed_data()
    categories = sorted(list(set([str(d['Categories']).strip() for d in data if d.get('Categories')])))
    return render_template('index.html', categories=categories)

@app.route('/api/data/<category_slug>')
def get_category_data(category_slug):
    all_data = get_processed_data()
    slug = category_slug.strip().lower()
    if slug == 'all':
        return jsonify({'status': 'success', 'data': all_data})

    filtered = [d for d in all_data if str(d['Categories']).strip().lower() == slug]
    return jsonify({'status': 'success', 'data': filtered})

# --- NAYA ROUTE: SMART SEARCH ---
@app.route('/api/smart_search', methods=['POST'])
def smart_search():
    data = request.get_json()
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({'keywords': [], 'translated_phrase': ''})

    system_prompt = """
    You are an e-commerce search assistant. Translate the following Roman Urdu/Hindi text into simple English search keywords. 
    Provide ONLY the English translation without any extra words or quotes.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=30
        )
        english_translation = chat_completion.choices[0].message.content.strip(' "')
        
        text = english_translation.lower()
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        cleaned_tokens = [t for t in tokens if t not in stop_words]
        
        return jsonify({
            'keywords': cleaned_tokens, 
            'translated_phrase': " ".join(cleaned_tokens)
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'keywords': user_query.lower().split(), 'translated_phrase': user_query})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)