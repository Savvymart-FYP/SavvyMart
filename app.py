import pandas as pd
import os
from flask import Flask, render_template, jsonify
import numpy as np
 app = 4
app = Flask(__name__)

# Path setup
BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, 'Websites Data', 'data.csv')


def get_processed_data():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        df = pd.read_csv(DATA_FILE)

        # Reviews grouping with NaN fix
        review_cols = ['Reviewer Name', 'Rating', 'Review Body', 'Date']
        reviews_dict = df.groupby('Product URL').apply(
            lambda x: x[review_cols].replace({np.nan: None}).to_dict('records')
        ).to_dict()

        df_unique = df.groupby('Product URL', as_index=False).first()

        # Numeric Clean-up
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

        return df_unique.replace({np.nan: None}).to_dict('records')
    except Exception as e:
        print(f"Server Error: {e}")
        return []


@app.route('/')
def home():
    data = get_processed_data()
    # Categories list generate karna
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)