import pandas as pd
import os
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

BASE_DIR  = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, 'Websites Data', 'data.csv')

# ── Semantic Search load ───────────────────────────────────────
try:
    from search_engine import SearchEngine, normalize_query
    search_engine = SearchEngine()
    SEARCH_READY  = True
    print("  Semantic Search: READY")
except Exception as e:
    search_engine = None
    SEARCH_READY  = False
    normalize_query = lambda x: x
    print(f"  Semantic Search: OFF — {e}")


def get_processed_data():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        df = pd.read_csv(DATA_FILE)
        review_cols  = ['Reviewer Name', 'Rating', 'Review Body', 'Date']
        reviews_dict = (
            df.groupby('Product URL')[review_cols]
            .apply(lambda x: x.replace({np.nan: None}).to_dict('records'))
            .to_dict()
        )
        df_unique = df.groupby('Product URL', as_index=False).first()
        df_unique['Rating']         = pd.to_numeric(df_unique['Rating'], errors='coerce').fillna(4.0)
        df_unique['Regular Price']  = pd.to_numeric(df_unique['Regular Price'], errors='coerce').fillna(0)
        df_unique['Sale Price']     = pd.to_numeric(df_unique['Sale Price'], errors='coerce').fillna(0)
        df_unique['All_Reviews']    = df_unique['Product URL'].map(reviews_dict)
        df_unique['Review_Count']   = df_unique['All_Reviews'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        def calc_metrics(row):
            reg, sale = row['Regular Price'], row['Sale Price']
            discount  = round(((reg - sale) / reg) * 100) if reg > sale > 0 else 0
            savings   = reg - sale if reg > sale else 0
            return pd.Series([discount, savings])

        df_unique[['Discount_Pct', 'Savings']] = df_unique.apply(calc_metrics, axis=1)
        return df_unique.replace({np.nan: None}).to_dict('records')
    except Exception as e:
        print(f"Server Error: {e}")
        return []


# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def home():
    data       = get_processed_data()
    categories = sorted(list(set([
        str(d['Categories']).strip() for d in data if d.get('Categories')
    ])))
    return render_template('index.html', categories=categories)


# ── Search Test Page ─────────────────────────────────────────
# http://localhost:5000/test  ← yahan se kholo
@app.route('/test')
def search_test():
    return send_from_directory('.', 'search_test.html')


@app.route('/api/data/<category_slug>')
def get_category_data(category_slug):
    all_data = get_processed_data()
    slug     = category_slug.strip().lower()
    if slug == 'all':
        return jsonify({'status': 'success', 'data': all_data})
    filtered = [d for d in all_data if str(d['Categories']).strip().lower() == slug]
    return jsonify({'status': 'success', 'data': filtered})


@app.route('/api/search')
def semantic_search():
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 20))

    if not query:
        return jsonify({'status': 'success', 'urls': [], 'mode': 'empty'})

    normalized = normalize_query(query)

    if SEARCH_READY:
        try:
            results = search_engine.search(query, top_k=limit)
            urls    = [r['Product URL'] for r in results]
            scores  = {r['Product URL']: r['search_score'] for r in results}
            return jsonify({
                'status'    : 'success',
                'urls'      : urls,
                'scores'    : scores,
                'mode'      : 'semantic',
                'normalized': normalized,
                'count'     : len(urls),
            })
        except Exception as e:
            print(f"Search error: {e}")

    all_data = get_processed_data()
    matched  = [d['Product URL'] for d in all_data if query.lower() in str(d.get('Product Name', '')).lower()]
    return jsonify({
        'status'    : 'success',
        'urls'      : matched,
        'mode'      : 'keyword',
        'normalized': normalized,
        'count'     : len(matched),
    })


@app.route('/health')
def health():
    return jsonify({
        'status'          : 'ok',
        'search_ready'    : SEARCH_READY,
        'products_indexed': len(search_engine.products) if SEARCH_READY else 0,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# http://localhost:5000/test
