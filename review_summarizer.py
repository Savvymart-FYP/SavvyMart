import os
from groq import Groq
from flask import Blueprint, jsonify, request

summarizer_bp = Blueprint('summarizer', __name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)


# chalo tumharay kehne par main yeh data Reviews summarization main nhi handle karwata 
JUNK_PATTERNS = [
    'review via', 'via instagram', 'via whatsapp', 'via facebook',
    'via twitter', 'via social', 'instagram review', 'whatsapp review',
]

def is_meaningful_review(text: str) -> bool:
    t = text.lower().strip()
    if len(t) < 15:
        return False
    if any(pat in t for pat in JUNK_PATTERNS):
        return False
    if len(t.split()) < 3:
        return False
    return True


def get_product_summary(all_reviews: list) -> dict:
    if not all_reviews:
        return {"success": False, "summary": "No reviews found for this product.", "review_count": 0}

    review_texts = []
    ratings_only = []

    for r in all_reviews:
        name   = r.get('Reviewer Name', 'Anonymous') or 'Anonymous'
        rating = r.get('Rating', 'N/A')
        body   = (r.get('Review Body', '') or '').strip()

        if is_meaningful_review(body):
            review_texts.append(f"- {name} (Rating: {rating}/5): {body}")
        else:
            try:
                ratings_only.append(float(rating))
            except (TypeError, ValueError):
                pass

    if not review_texts:
        total = len(all_reviews)
        if ratings_only:
            avg = round(sum(ratings_only) / len(ratings_only), 1)
            return {
                "success": True,
                "summary": (
                    f"This product has {total} ratings with an average of {avg}/5, "
                    f"but no written reviews are available — customers shared feedback "
                    f"via social media (Instagram/WhatsApp) without leaving text reviews."
                ),
                "review_count": total,
                "no_text_reviews": True
            }
        return {"success": False, "summary": "No written reviews found for this product.", "review_count": 0}

    combined_reviews = "\n".join(review_texts)

# Note acha tum nhi chahatay roman urdu trnasaltion kiun ke us main HINDI words hain because Llama 3.3 model ziada tar hindi par train hai

    # =========================================================
    # PROMPT IS 100% ENGLISH — This is intentional and critical.
    # Llama 3.3 has far more Hindi training data than Urdu.
    # Any Roman Urdu/Hindi in the prompt causes Hindi output
    # e.g. "darshata hai", "pata chalta hai", "sankit" etc.
    # Solution: pure English prompt + explicit language rule.
    # =========================================================
    system_msg = """You are a product review summarizer for SavvyMart, a Pakistani ecommerce platform.

STRICT RULES — follow exactly:
1. ALWAYS write your summary in ENGLISH only. No exceptions.
2. Do NOT use Hindi, Urdu, Roman Urdu, or any other language. English only.
3. Write 2-4 natural sentences. No bullet points, no headers, no numbering.
4. Cover: overall customer sentiment, what customers liked, any complaints or issues.
5. Be concise and helpful for a shopper deciding whether to buy this product."""

    user_msg = f"""Summarize these customer reviews in ENGLISH only:

{combined_reviews}

Remember: English only. Natural sentences. No bullet points."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=300,
            temperature=0.3
        )

        summary = response.choices[0].message.content.strip()
        return {"success": True, "summary": summary, "review_count": len(all_reviews)}

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower():
            return {"success": False, "summary": "Groq API key invalid. Check your .env file.", "review_count": len(all_reviews)}
        elif "429" in error_msg or "rate" in error_msg.lower():
            return {"success": False, "summary": "Rate limit reached. Please try again shortly.", "review_count": len(all_reviews), "retry": True}
        else:
            return {"success": False, "summary": f"Could not generate summary: {error_msg}", "review_count": len(all_reviews)}


@summarizer_bp.route('/api/summarize', methods=['POST'])
def summarize_reviews():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "summary": "Invalid request data."}), 400
        reviews = data.get('reviews', [])
        if not reviews:
            return jsonify({"success": False, "summary": "No reviews for this product.", "review_count": 0})
        result = get_product_summary(reviews)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "summary": f"Server error: {str(e)}"}), 500
