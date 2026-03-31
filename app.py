import pandas as pd
import os
import pickle
from flask import Flask, render_template, jsonify, request
import numpy as np
from NLP_Pipeline.sentiment_score import get_sentiment_percentages

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, 'Websites Data', 'data.csv')
CACHE_FILE = os.path.join(BASE_DIR, 'Websites Data', 'processed_data.pkl')


def pre_process_everything():
    """
    Processes the raw CSV data, calculates average ratings for products,
    pre-calculates sentiments, and saves everything to a Pickle cache.
    """
    if not os.path.exists(DATA_FILE):
        return None

    print("Generating Fast Cache... (Calculating Average Ratings & Sentiments)")
    df = pd.read_csv(DATA_FILE)

    if 'Fraud_Label' not in df.columns:
        df['Fraud_Label'] = 'Unknown'

    # Convert Rating to numeric for calculation
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(4.0)

    # --- NEW: Calculate Average Rating for each Product ---
    # This groups by URL and finds the mean rating of all reviews for that specific product
    avg_ratings = df.groupby('Product URL')['Rating'].mean().round(1).to_dict()

    # Group reviews to show in the modal
    review_cols = ['Reviewer Name', 'Rating', 'Review Body', 'Date', 'Fraud_Label']
    reviews_dict = df.groupby('Product URL', group_keys=False).apply(
        lambda x: x[review_cols].replace({np.nan: None}).to_dict('records')
    ).to_dict()

    # Get unique products
    df_unique = df.groupby('Product URL', as_index=False).first()

    # Assign the calculated Average Rating instead of the first review's rating
    df_unique['Rating'] = df_unique['Product URL'].map(avg_ratings)

    # Pre-calculate prices and basic info
    df_unique['Regular Price'] = pd.to_numeric(df_unique['Regular Price'], errors='coerce').fillna(0)
    df_unique['Sale Price'] = pd.to_numeric(df_unique['Sale Price'], errors='coerce').fillna(0)

    # Pre-calculate Sentiments (Heavy process, done once)
    print("Analyzing Sentiments...")
    df_unique['All_Reviews'] = df_unique['Product URL'].map(reviews_dict)
    df_unique[['Pos_Pct', 'Neg_Pct', 'Neu_Pct']] = df_unique['All_Reviews'].apply(get_sentiment_percentages)
    df_unique[['Pos_Pct', 'Neg_Pct', 'Neu_Pct']] = df_unique[['Pos_Pct', 'Neg_Pct', 'Neu_Pct']].fillna(0)
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
    """
    Slices the data for pagination and provides all available brands for the filters.
    """
    try:
        df_work = GLOBAL_DF.copy()

        if category_slug.lower() != 'all':
            df_work = df_work[df_work['Categories'].astype(str).str.strip().str.lower() == category_slug.lower()]

        # Get ALL unique brands for this category (Used for the Left Sidebar)
        all_brands = sorted(df_work['Brand Name'].dropna().unique().tolist())

        total_items = len(df_work)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        df_chunk = df_work.iloc[start_idx:end_idx].copy()
        has_next = end_idx < total_items

        if df_chunk.empty:
            return [], False, all_brands

        # Dynamic metrics
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


if __name__ == '__main__':
    app.run(debug=True)