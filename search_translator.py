import os
from groq import Groq
from flask import Blueprint, jsonify, request

# =====================================================================
# SavvyMart - Multilingual Search Translation Module
# Pure LLM approach — no hardcoded word lists
# Roman Urdu / Urdu / English / Mixed → English + Keywords
# =====================================================================

translator_bp = Blueprint('translator', __name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# yeh acha Idea hai BRO!!! hum Cache kar lete hain taakay baar baar 1 product ke liye Reviews summarize na hon aur API hit na ho

# --- IN-MEMORY CACHE (same session mein same query dobara API nahi jayegi) ---
translation_cache = {}


def translate_to_english(query: str) -> dict:
    """
    Koi bhi query lo — Roman Urdu, Urdu script, English, Mixed, Unknown words —
    LLM khud samjhe ga aur English + keywords return karega.
    """

    query = query.strip()

    if not query:
        return {
            "success": False,
            "translated": "",
            "keywords": [],
            "original": query,
            "was_translated": False
        }

    # Cache check — same query dobara API call nahi hogi
    cache_key = query.lower()
    if cache_key in translation_cache:
        print(f"[CACHE HIT]: '{query}'")
        return translation_cache[cache_key]

    # =====================================================================
    # PURE LLM PROMPT — No hardcoded rules, model khud decide kare ga
    # =====================================================================
    prompt = f"""You are a smart search assistant for SavvyMart — a Pakistani ecommerce platform similar to Daraz.

A user has typed this in the search bar: "{query}"

The user could have typed in:
- Roman Urdu (e.g. "kala batwa", "balon ka tel", "raat wali cream")
- Urdu script (e.g. "کالا بٹوہ")
- English (e.g. "black wallet", "hair oil")
- Mixed language (e.g. "white parse", "ladies ka bag")
- Completely unknown or misspelled words — use context to understand

YOUR TASK:
1. Understand what product the user is looking for
2. Translate/interpret it into a clear English product phrase (1-5 words)
3. Generate 6-8 relevant ecommerce search keywords that would help find this product

IMPORTANT RULES:
- If already English → keep as-is, still generate keywords
- Use your language understanding — do NOT rely on a fixed dictionary
- For unknown words → use context clues from surrounding words
- Think like a Pakistani shopper — what product are they looking for?
- Keywords should be single words that appear in product names/descriptions

RESPOND IN THIS EXACT FORMAT (2 lines only, nothing else):
TRANSLATION: <english product phrase>
KEYWORDS: <word1, word2, word3, word4, word5, word6, word7, word8>

EXAMPLES:
Input: "kala batwa"
TRANSLATION: black wallet
KEYWORDS: wallet, black, leather, purse, bifold, mens, slim, cardholder

Input: "balon ka tel"
TRANSLATION: hair oil
KEYWORDS: hair, oil, argan, biotin, serum, scalp, growth, treatment

Input: "raat ko sonay wali cream"
TRANSLATION: night sleeping cream
KEYWORDS: night, cream, sleeping, serum, moisturizer, skin, whitening, anti-aging

Input: "safaid wallet"
TRANSLATION: white wallet
KEYWORDS: wallet, white, purse, leather, clutch, ladies, slim, cardholder

Input: "aurton ka batwa"
TRANSLATION: women wallet
KEYWORDS: wallet, women, ladies, purse, clutch, handbag, female, girls

Input: "sasta mobile"
TRANSLATION: budget smartphone
KEYWORDS: mobile, phone, smartphone, android, budget, cheap, affordable, entry-level

Input: "surmai rang ka joota"
TRANSLATION: grey shoes
KEYWORDS: shoes, grey, gray, sneakers, casual, sport, men, leather

Input: "bachon ka shampoo"
TRANSLATION: kids shampoo
KEYWORDS: shampoo, kids, baby, children, gentle, hair, wash, tear-free

Now respond ONLY for: "{query}"
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1
        )

        raw = response.choices[0].message.content.strip()
        print(f"[TRANSLATION RAW] Query='{query}' | Raw='{raw}'")

        translated = query  # fallback
        keywords = []

        for line in raw.split('\n'):
            line = line.strip()
            if line.upper().startswith('TRANSLATION:'):
                translated = line.split(':', 1)[1].strip().strip('"\'')
            elif line.upper().startswith('KEYWORDS:'):
                kw_raw = line.split(':', 1)[1].strip()
                keywords = [k.strip().lower() for k in kw_raw.split(',') if k.strip()]

        was_translated = translated.lower() != query.lower()

        result = {
            "success": True,
            "translated": translated,
            "keywords": keywords[:8],
            "original": query,
            "was_translated": was_translated
        }

        translation_cache[cache_key] = result
        print(f"[TRANSLATED] '{query}' → '{translated}' | keywords: {keywords}")

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"[TRANSLATION ERROR] {error_msg}")

        fallback = {
            "success": False,
            "translated": query,
            "keywords": [],
            "original": query,
            "was_translated": False
        }

        if "401" in error_msg or "authentication" in error_msg.lower():
            fallback["error"] = "Groq API key galat hai"
        elif "429" in error_msg or "rate" in error_msg.lower():
            fallback["error"] = "Rate limit"
        else:
            fallback["error"] = error_msg

        return fallback


# =====================================================================
# FLASK ROUTE
# =====================================================================
@translator_bp.route('/api/translate-search', methods=['POST'])
def translate_search():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "translated": "", "error": "Invalid request"}), 400

        query = data.get('query', '').strip()
        if not query:
            return jsonify({"success": False, "translated": "", "error": "Empty query"}), 400

        result = translate_to_english(query)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "translated": "",
            "error": f"Server error: {str(e)}"
        }), 500
