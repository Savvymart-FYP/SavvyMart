"""
============================================================
  build_embeddings.py  —  IMPROVED VERSION
  
  Improvement: Product text mein Roman Urdu keywords add kiye
  Taake "pyaaz wala shampoo" → "Onion Shampoo" match ho
  
  Run: python build_embeddings.py
============================================================
"""
import os, re, pickle
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_CSV       = os.path.join(BASE_DIR, 'Websites Data', 'data.csv')
OUTPUT_PKL     = os.path.join(BASE_DIR, 'product_embeddings.pkl')
MODEL_PATH     = os.path.join(BASE_DIR, 'savvymart_model', 'final')
FALLBACK_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def clean_text(t):
    return re.sub(r'[^\x00-\x7F]+', '', str(t)).strip()


# ─────────────────────────────────────────────────────────────
#  ROMAN URDU KEYWORDS — Har product ke liye
#  Yeh keywords product embedding text mein add honge
#  Taake user jab bhi koi variation type kare, product mile
#
#  KEY:   product name ka unique part (lowercase)
#  VALUE: roman urdu + english synonyms/variations
# ─────────────────────────────────────────────────────────────
PRODUCT_RU_KEYWORDS = {

    # ── HAIR OIL / BEAUTY (hsBeauty) ──────────────────────────────
    "onion & ginger shampoo": (
        "pyaaz shampoo pyaaz wala shampoo pyaaz ki shampoo pyaaz ka shampoo "
        "baal girna rokne wala shampoo hair fall shampoo onion shampoo "
        "baal majboot shampoo khoobsurat baal pyaz shampoo"
    ),
    "hair growth oil": (
        "baal ugana baal badhana baal lamba karna hair growth oil "
        "baal girna band karo baal jharne wala tel hair fall oil "
        "baal ka tel balo ka tel"
    ),
    "neem & alovera facewash": (
        "neem facewash aloe vera facewash muhna saaf karna face wash "
        "chehra dhona neem wala facewash muhna dhone wala sabun "
        "face cleaner skin cleanser"
    ),
    "hyaluronic acid serum": (
        "face serum chehra serum skin serum "
        "glowing skin moisturizing serum hyaluronic "
        "face ka serum chehra chamkana"
    ),
    "vitamin c serum": (
        "vitamin c serum chehra chamkana glowing skin "
        "vitamin c wala serum bright skin face serum "
        "daag dhabbe hatana whitening serum"
    ),
    "night face serum": (
        "raat ka serum night serum so kar uthne wala serum "
        "raat ko lagane wala serum night cream serum "
        "sone se pehle face serum"
    ),
    "body therapist moisturizing cream": (
        "body cream moisturizer lotion jism ki cream "
        "rukha pan duur karna dry skin cream body lotion "
        "naram skin cream moisturizing"
    ),
    "hair serum": (
        "baal serum hair serum baal chamkana silky hair "
        "baal mulayam baal serum balo ko chamkana "
        "smooth hair serum"
    ),

    # ── BAGS (Borsa) ──────────────────────────────────────────────
    "leather duffle": (
        "leather bag safar ka bag travel bag gym bag "
        "bada bag duffle bag weekend bag journey bag "
        "safar wala bag leather duffle"
    ),
    "crossbody bag": (
        "crossbody bag shoulder bag sling bag "
        "kandhe ka bag chhota bag daily use bag "
        "cross body ek tarfa bag"
    ),
    "chest & crossbody": (
        "chest bag crossbody shoulder bag sling "
        "kandhe wala bag chhota bag "
    ),
    "laptop convertible backpack": (
        "laptop bag backpack laptop wala bag office bag "
        "laptop carry bag 2 in 1 bag convertible "
        "laptop aur backpack"
    ),
    "premium leather backpack": (
        "backpack leather backpack school bag college bag "
        "laptop bag office bag peethe wala bag "
        "anti theft lock backpack"
    ),
    "premium travel & organizer backpack": (
        "safar ka bag travel bag bada backpack organizer "
        "journey bag trip bag travel organizer "
        "safar wala bada bag"
    ),
    "mini leather duffle": (
        "chota bag mini bag chhota leather bag "
        "small duffle weekend bag mini duffle "
        "choti size bag"
    ),
    "fannypack": (
        "waist bag fanny pack belt bag kamar ka bag "
        "chhota waist bag pouch bag "
    ),
    "signature (black)": (
        "kala bag black bag kali bag signature black "
        "kala rango ka bag"
    ),
    "pitch black": (
        "kala bag black bag kali bag pitch black "
        "all black bag kala rango ka bag"
    ),
    "premium black quilted": (
        "kala bag black bag quilted duffle kala duffle "
        "black premium bag"
    ),
    "signature (navy)": (
        "navy blue bag nila bag dark blue bag "
        "navy color bag neela bag"
    ),
    "signature (red wine)": (
        "lal bag dark red bag wine color bag maroon bag "
        "lal rango ka bag"
    ),
    "signature (chocolate)": (
        "brown bag chocolate color bag brownish bag "
        "bhura rang ka bag"
    ),
    "toiletry bag": (
        "toiletry bag travel pouch bathroom bag "
        "safar mein makeup bag chhota pouch "
        "washroom ka bag travel kit"
    ),
    "sling 2.0": (
        "sling bag shoulder bag chhota bag casual bag "
        "ek kandhe ka bag sling crossbody"
    ),
    "laptop/macbook/ipad bag": (
        "laptop bag macbook bag ipad bag office bag "
        "premium laptop carry bag "
    ),
    "garment duffle": (
        "kapron ka bag garment bag suit bag "
        "travel clothes bag kapray rakhne ka bag "
        "suit cover bag"
    ),
    "harper": (
        "leather bag shoulder bag premium bag "
        "leather shoulder sling harper"
    ),

    # ── ELECTRONICS (Zero Lifestyle — earphones/headphones) ──────
    "rover pro": (
        "earphones sasti earphones budget earphones "
        "wired earphones kaan wali earphones "
        "zero lifestyle earphones"
    ),
    "luna": (
        "wireless earbuds bluetooth earbuds "
        "wireless earphones bluetooth earphones "
        "wire less earbuds"
    ),
    "wave pro": (
        "wireless earbuds bluetooth earphones "
        "premium earbuds wave pro"
    ),
    "wave ": (  # note space to avoid matching wave pro
        "earphones wireless earbuds bluetooth "
        "sasti wireless earphones"
    ),
    "carbon": (
        "gaming earphones sporty earphones "
        "active earphones carbon"
    ),
    "ignite": (
        "wireless earbuds bluetooth gaming earphones "
        "premium wireless ignite"
    ),
    "flair": (
        "earphones sasti earphones colorful earphones "
        "flair earphones budget"
    ),
    "gravity": (
        "earphones sport earphones gravity "
        "workout earphones"
    ),
    "z 811": (
        "earphones wired earphones z811 "
        "basic earphones"
    ),
    "zero aura": (
        "earphones wireless budget earbuds "
        "zero aura sasti earbuds"
    ),
    "zero arcade": (
        "gaming earphones zero arcade "
        "earphones gaming"
    ),
    "quantum": (
        "wireless earbuds premium earbuds "
        "quantum bluetooth earbuds"
    ),
    "icon": (
        "premium earbuds wireless earphones "
        "icon bluetooth earbuds"
    ),
    "storm headphones": (
        "headphones gaming headphones over ear "
        "sarnay wale headphones bade headphones "
        "wired headphones gaming"
    ),
    "orbit 2": (
        "headphones over ear headphones premium "
        "noise cancel headphones orbit"
    ),
    "zero armour": (
        "noise cancel earphones ANC earphones "
        "shor band karne wali earphones "
        "premium noise cancelling zero armour"
    ),
    "display": (
        "display earbuds screen wali earbuds "
        "digital display earphones"
    ),
    "revoltt": (
        "premium earbuds revoltt wireless "
        "bluetooth earbuds revoltt"
    ),
    "revoltt pro": (
        "premium earbuds revoltt pro wireless "
        "top earbuds"
    ),
    "crown": (
        "premium earbuds crown wireless "
        "high end earbuds"
    ),
    "jewel": (
        "premium earbuds jewel wireless "
        "luxury earbuds"
    ),
    "regal ai": (
        "AI earbuds smart earbuds regal ai "
        "artificial intelligence earphones "
        "smart noise cancel"
    ),
    "nebula": (
        "wireless earbuds nebula bluetooth "
        "premium earbuds"
    ),
    "lunar 360": (
        "360 sound earbuds premium earbuds "
        "lunar wireless high end"
    ),
}


def get_ru_keywords(product_name: str) -> str:
    """Product name ke liye Roman Urdu keywords dhundho."""
    name_lower = product_name.lower()
    matched    = []

    for key, keywords in PRODUCT_RU_KEYWORDS.items():
        if key in name_lower:
            matched.append(keywords.strip())

    return " ".join(matched)


def build():
    print("=" * 60)
    print("  SavvyMart — Building IMPROVED Product Embeddings")
    print("=" * 60)

    model_to_use = MODEL_PATH if Path(MODEL_PATH).exists() else FALLBACK_MODEL
    status = "Fine-tuned ✓" if Path(MODEL_PATH).exists() else "Base model"
    print(f"\n  Model  : {status}")

    df            = pd.read_csv(DATA_CSV)
    avg_ratings   = df.groupby('Product URL')['Rating'].mean().round(1)
    review_counts = df.groupby('Product URL')['Review ID'].count()
    df_u          = df.groupby('Product URL', as_index=False).first()
    df_u['avg_rating']   = df_u['Product URL'].map(avg_ratings)
    df_u['review_count'] = df_u['Product URL'].map(review_counts)

    products = []
    for _, row in df_u.iterrows():
        products.append({
            'Product Name'  : str(row.get('Product Name', '')),
            'clean_name'    : clean_text(row.get('Product Name', '')),
            'Product URL'   : str(row.get('Product URL', '')),
            'Image URL'     : str(row.get('Image URL', '')),
            'Sale Price'    : float(row.get('Sale Price', 0) or 0),
            'Regular Price' : float(row.get('Regular Price', 0) or 0),
            'Categories'    : str(row.get('Categories', '')),
            'Brand Name'    : str(row.get('Brand Name', '')),
            'avg_rating'    : round(float(row.get('avg_rating') or 4.0), 1),
            'review_count'  : int(row.get('review_count') or 0),
        })

    print(f"  Products: {len(products)}")

    # ── IMPROVED embedding text ──────────────────────────────
    # OLD: "Onion & Ginger Shampoo Hair Oil"
    # NEW: "Onion & Ginger Shampoo Hair Oil pyaaz shampoo pyaaz wala..."
    embed_texts = []
    matched_count = 0
    for p in products:
        ru_kw  = get_ru_keywords(p['clean_name'])
        text   = f"{p['clean_name']} {p['Categories']}"
        if ru_kw:
            text += f" {ru_kw}"
            matched_count += 1
        embed_texts.append(text)

    print(f"  RU keywords added: {matched_count}/{len(products)} products")
    print(f"\n  Sample improved texts:")
    for i, (p, t) in enumerate(zip(products, embed_texts)):
        if 'Onion' in p['clean_name'] or 'Hair Growth' in p['clean_name'] or 'Signature (Black)' in p['clean_name']:
            print(f"\n  Product: {p['clean_name']}")
            print(f"  Text   : {t[:120]}...")

    # ── Encode ───────────────────────────────────────────────
    print(f"\n  Encoding {len(embed_texts)} products...")
    model      = SentenceTransformer(model_to_use)
    embeddings = model.encode(embed_texts, show_progress_bar=True, batch_size=32)

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump({
            'embeddings'  : embeddings,
            'products'    : products,
            'embed_texts' : embed_texts,   # debug ke liye save
        }, f)

    print(f"\n  Saved  : product_embeddings.pkl  shape={embeddings.shape}")
    print(f"  Next   : python app.py")
    print("=" * 60)


if __name__ == '__main__':
    build()
