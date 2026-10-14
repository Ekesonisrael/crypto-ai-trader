import requests

# ==========================================
# CRYPTOCURRENCY.CV API WRAPPER (No Key Required)
# ==========================================

def get_market_sentiment():
    """Fetches the overall AI-analyzed market sentiment."""
    try:
        response = requests.get("https://cryptocurrency.cv/api/sentiment", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "score": data.get("score", 0),
                "label": data.get("label", "neutral"),
                "confidence": data.get("confidence", 0)
            }
    except Exception as e:
        print(f"News Agent Error (Sentiment): {e}")
    return {"score": 0, "label": "neutral", "confidence": 0}

def get_x_sentiment():
    """Fetches sentiment specifically from X (Twitter)."""
    try:
        response = requests.get("https://cryptocurrency.cv/api/social/x/sentiment", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "score": data.get("score", 0),
                "volume": data.get("volume", 0)
            }
    except Exception as e:
        print(f"News Agent Error (X): {e}")
    return {"score": 0, "volume": 0}

def get_breaking_news():
    """Fetches the latest breaking news headlines."""
    try:
        response = requests.get("https://cryptocurrency.cv/api/breaking", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("articles", [])
    except Exception as e:
        print(f"News Agent Error (Breaking): {e}")
    return []

def check_news_impact():
    """
    Checks the news and returns a status:
    - 'VETO': Major negative news (Hack, SEC, etc.)
    - 'BULLISH': Strong positive sentiment
    - 'BEARISH': Strong negative sentiment
    - 'NEUTRAL': Noise / No strong signal
    """
    sentiment = get_market_sentiment()
    breaking = get_breaking_news()
    
    # Check for Tier 1 Veto Keywords
    veto_keywords = ["HACK", "SEC", "BAN", "BANKRUPTCY", "CRASH", "FED", "WAR"]
    for article in breaking:
        title = article.get("title", "").upper()
        if any(keyword in title for keyword in veto_keywords):
            return "VETO", f"Breaking News: {article.get('title')}"
            
    # Check Sentiment Score
    score = sentiment['score']
    if score > 0.5:
        return "BULLISH", f"Market Sentiment Bullish ({score})"
    elif score < -0.5:
        return "BEARISH", f"Market Sentiment Bearish ({score})"
        
    return "NEUTRAL", "No strong fundamental signals."
