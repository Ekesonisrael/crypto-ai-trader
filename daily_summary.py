import requests
import json
from datetime import datetime

def fetch_fundamentals():
    print("📰 Fetching fundamental data...")
    
    state = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral",
        "x_sentiment": 0.0,
        "breaking_news": [],
        "critical_alert": False,
        "alert_reason": ""
    }
    
    try:
        r1 = requests.get("https://cryptocurrency.cv/api/sentiment", timeout=10)
        if r1.status_code == 200:
            data = r1.json()
            state["sentiment_score"] = data.get("score", 0.0)
            state["sentiment_label"] = data.get("label", "Neutral")
            
        r2 = requests.get("https://cryptocurrency.cv/api/social/x/sentiment", timeout=10)
        if r2.status_code == 200:
            data = r2.json()
            state["x_sentiment"] = data.get("score", 0.0)
            
        r3 = requests.get("https://cryptocurrency.cv/api/breaking", timeout=10)
        if r3.status_code == 200:
            data = r3.json()
            news_items = data.get("articles", [])
            state["breaking_news"] = [item.get("title") for item in news_items[:3]]
            
            critical_keywords = ["HACK", "SEC", "EXPLOIT", "BAN", "EMERGENCY", "CRASH"]
            for title in state["breaking_news"]:
                for kw in critical_keywords:
                    if kw in title.upper():
                        state["critical_alert"] = True
                        state["alert_reason"] = f"Critical news: {title}"
                        break
                if state["critical_alert"]: break
                        
    except Exception as e:
        print(f"Error fetching fundamentals: {e}")
        
    with open('fundamental_state.json', 'w') as f:
        json.dump(state, f, indent=4)
        
    print(f"✅ Fundamentals saved. Sentiment: {state['sentiment_label']}")

if __name__ == "__main__":
    fetch_fundamentals()
