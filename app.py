import pandas as pd
import os
import pickle
from flask import Flask, render_template, jsonify, request
import numpy as np
from dotenv import load_dotenv
load_dotenv()   # ← .env file read karega jo tum ne API key yahan daali hai

# --- Review Summarization Module ---
from review_summarizer import summarizer_bp

# --- Multilingual Search Translation Module ---
from search_translator import translator_bp

# NOTE: NLP_Pipeline (sentiment_score) temporarily removed. taakay tumhen Dependencies na aayen yahan
# Jab teammate git par push kare tab yeh line uncomment karna:
# from NLP_Pipeline.sentiment_score import get_sentiment_percentages

app = Flask(__name__)
app.register_blueprint(summarizer_bp)
app.register_blueprint(translator_bp)   # ← Translation routes register

BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, 'Websites Data', 'data.csv')
CACHE_FILE = os.path.join(BASE_DIR, 'Websites Data', 'processed_data.pkl')


def pre_process_everything():
    if not os.path.exists(DATA_FILE):
        return None

    print("Generating Fast Cache...")
    df = pd.read_csv(DATA_FILE)

    if 'Fraud_Label' not in df.columns:
        df['Fraud_Label'] = 'Unknown'

    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(4.0)

    avg_ratings = df.groupby('Product URL')['Rating'].mean().round(1).to_dict()

    review_cols = ['Reviewer Name', 'Rating', 'Review Body', 'Date', 'Fraud_Label']
    reviews_dict = df.groupby('Product URL', group_keys=False).apply(
        lambda x: x[review_cols].replace({np.nan: None}).to_dict('records')
    ).to_dict()

    df_unique = df.groupby('Product URL', as_index=False).first()
    df_unique['Rating'] = df_unique['Product URL'].map(avg_ratings)

    df_unique['Regular Price'] = pd.to_numeric(df_unique['Regular Price'], errors='coerce').fillna(0)
    df_unique['Sale Price'] = pd.to_numeric(df_unique['Sale Price'], errors='coerce').fillna(0)

    # Sentiment temporarily disabled — 0 set kar rahe hain
    df_unique['All_Reviews'] = df_unique['Product URL'].map(reviews_dict)
    df_unique['Pos_Pct'] = 0
    df_unique['Neg_Pct'] = 0
    df_unique['Neu_Pct'] = 0
    df_unique['Review_Count'] = df_unique['All_Reviews'].apply(len)

    processed_data = {
        'df': df_unique,
        'categories': sorted(list(set([str(c).strip() for c in df_unique['Categories'].dropna() if c])))
    }

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(processed_data, f)

    return processed_data


# Initialize Global Data
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        cache_data = pickle.load(f)
else:
    cache_data = pre_process_everything()

GLOBAL_DF = cache_data['df']
GLOBAL_CATEGORIES = cache_data['categories']


def get_processed_data(category_slug='all', page=1, limit=8):
    try:
        df_work = GLOBAL_DF.copy()

        if category_slug.lower() != 'all':
            df_work = df_work[df_work['Categories'].astype(str).str.strip().str.lower() == category_slug.lower()]

        all_brands = sorted(df_work['Brand Name'].dropna().unique().tolist())

        total_items = len(df_work)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        df_chunk = df_work.iloc[start_idx:end_idx].copy()
        has_next = end_idx < total_items

        if df_chunk.empty:
            return [], False, all_brands

        def calc_metrics(row):
            reg, sale = row['Regular Price'], row['Sale Price']
            discount = round(((reg - sale) / reg) * 100) if reg > sale > 0 else 0
            return pd.Series([discount, (reg - sale if reg > sale else 0)])

        df_chunk[['Discount_Pct', 'Savings']] = df_chunk.apply(calc_metrics, axis=1)

        return df_chunk.replace({np.nan: None}).to_dict('records'), has_next, all_brands
    except Exception as e:
        print(f"Server Error: {e}")
        return [], False, []


@app.route('/')
def home():
    return render_template('index.html', categories=GLOBAL_CATEGORIES)


@app.route('/api/data/<category_slug>')
def get_category_data(category_slug):
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 8))
    data, has_next, brands = get_processed_data(category_slug, page, limit)
    return jsonify({
        'status': 'success',
        'data': data,
        'has_next': has_next,
        'brands': brands
    })


# =====================================================================
# BACKEND SEARCH ROUTE — Poora dataset search karta hai (not just loaded)
# =====================================================================
@app.route('/api/search', methods=['POST'])
def search_products():
    """
    POST /api/search
    Body: { "translated": "hair oil", "keywords": ["hair","oil",...], "original": "balon ka tel" }
    Returns: Top matching products from FULL dataset
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error"}), 400

        translated = (data.get('translated') or '').lower().strip()
        keywords   = [k.lower().strip() for k in (data.get('keywords') or []) if k]
        original   = (data.get('original') or '').lower().strip()

        if not translated and not keywords:
            return jsonify({"status": "success", "data": []})

        df_work = GLOBAL_DF.copy()

        def score_product(row):
            name     = str(row.get('Product Name') or '').lower()
            category = str(row.get('Categories') or '').lower()
            brand    = str(row.get('Brand Name') or '').lower()
            combined = name + ' ' + category + ' ' + brand
            score = 0

            # 1. Exact full phrase match in name — highest priority
            if translated and translated in name:
                score += 100

            # 2. Exact full phrase match in category
            if translated and translated in category:
                score += 80

            # 3. All words of translated query match in combined
            if translated:
                words = [w for w in translated.split() if len(w) > 1]
                if words:
                    all_match = all(w in combined for w in words)
                    if all_match:
                        score += 60
                    # Any word match
                    any_matches = sum(1 for w in words if w in combined)
                    score += any_matches * 20

            # 4. Keyword matches
            for kw in keywords:
                if kw and kw in combined:
                    score += 15

            # 5. Original query words (Roman Urdu fallback)
            if original:
                orig_words = [w for w in original.split() if len(w) > 2]
                for w in orig_words:
                    if w in combined:
                        score += 5

            return score

        df_work['_score'] = df_work.apply(score_product, axis=1)
        df_results = df_work[df_work['_score'] > 0].sort_values('_score', ascending=False)

        # Discount calculate karo
        def calc_discount(row):
            reg, sale = row['Regular Price'], row['Sale Price']
            return round(((reg - sale) / reg) * 100) if reg > sale > 0 else 0

        df_results = df_results.copy()
        df_results['Discount_Pct'] = df_results.apply(calc_discount, axis=1)
        df_results['Savings'] = df_results.apply(
            lambda r: r['Regular Price'] - r['Sale Price'] if r['Regular Price'] > r['Sale Price'] else 0, axis=1
        )
        df_results = df_results.drop(columns=['_score'])

        results = df_results.replace({np.nan: None}).to_dict('records')

        return jsonify({
            "status": "success",
            "data": results[:20],  # Top 20 results
            "total": len(results)
        })

    except Exception as e:
        print(f"Search Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
