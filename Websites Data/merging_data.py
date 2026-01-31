import pandas as pd
import os
import re


def clean_price(price_value):
    if pd.isna(price_value) or str(price_value).strip() == "":
        return 0

    # String mein convert kar ke faltu spaces hatana
    price_str = str(price_value).strip()

    # 1. Leading dots hatana (e.g., '.1,599.00' -> '1,599.00')
    price_str = price_str.lstrip('.')

    # 2. Commas hatana (e.g., '1,599.00' -> '1599.00')
    price_str = price_str.replace(',', '')

    # 3. Agar abhi bhi koi non-numeric character hai (except dot), usey hatana
    price_str = re.sub(r'[^\d.]', '', price_str)

    try:
        # Float mein convert kar ke round off karna (e.g., 1599.00 -> 1599)
        return float(price_str)
    except:
        return 0


def merge_website_data():
    files_config = {
        'Bags': '../Bags data/borsa_reviews_data.csv',
        'Electronics': '../Electronics data/ZeroLifestyle_reviews_data.csv',
        'Hair Oil': '../Hair Oil data/hsbeauty_reviews_data.csv'
    }

    columns_to_keep = [
        'Product Name', 'Product URL', 'Image URL', 'Sale Price',
        'Regular Price', 'Review ID', 'Reviewer Name', 'Rating',
        'Review Title', 'Review Body', 'Verified Buyer', 'Brand Name', 'Date'
    ]

    all_dfs = []

    print("Data merging, Date formatting aur Price cleaning start ho raha hai...")

    for category, file_path in files_config.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='latin1')
                df_filtered = df.reindex(columns=columns_to_keep)
                df_filtered['Categories'] = category

                # --- PRICE CLEANING LOGIC ---
                for col in ['Sale Price', 'Regular Price']:
                    if col in df_filtered.columns:
                        df_filtered[col] = df_filtered[col].apply(clean_price)

                # --- DATE FORMATTING LOGIC ---
                if 'Date' in df_filtered.columns:
                    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce', utc=True)
                    df_filtered['Date'] = df_filtered['Date'].dt.strftime('%Y-%m-%d')

                all_dfs.append(df_filtered)
                print(f"✅ {category} data processed (Prices & Dates cleaned).")
            except Exception as e:
                print(f"❌ Error processing {category}: {e}")
        else:
            print(f"⚠️ Warning: File nahi mili: {file_path}")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        output_file = 'data.csv'
        final_df.to_csv(output_file, index=False, encoding='utf-8')

        print("-" * 30)
        print(f"🎯 Perfect! '{output_file}' ab fully cleaned aur ready hai.")
        print(f"Sample Prices: {final_df['Sale Price'].head(3).tolist()}")
    else:
        print("Kuch bhi merge nahi ho saka.")


if __name__ == "__main__":
    merge_website_data()