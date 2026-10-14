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
        
    if os.path.isfile('fundamental_state.json'):
    with open('fundamental_state.json', 'r') as f:
        fund = json.load(f)

    message += "📰 *NEWS LAYER (Phase 4)*\n"
    message += f"API: `{fund.get('api_status', 'N/A')}`\n"
    message += f"Market Sentiment: `{fund['sentiment_label']}` ({fund['sentiment_score']:+.2f})\n"
    message += f"X/Twitter: `{fund['x_sentiment']:+.2f}`\n"

    # Show scored headlines
    scored = fund.get("headlines_scored", [])
    if scored:
        message += "\n*Top scored headlines:*\n"
        for h in scored[:3]:
            emoji = "🟢" if h['score'] > 0.2 else ("🔴" if h['score'] < -0.2 else "⚪")
            message += f"{emoji} {h['title'][:75]}...\n"
            message += f"   _{h['reason']}_ | {h['score']:+.2f}\n"

    # Active vetoes
    vetoes = fund.get("per_asset_vetoes", {})
    if vetoes:
        message += "\n🚫 *Active Vetoes:*\n"
        for asset, v in vetoes.items():
            message += f"• `{asset}`: {v['reason'][:60]}... (until {v['expires'][11:16]} UTC)\n"
    else:
        message += "\n✅ No active vetoes.\n" .dump(state, f, indent=4)
        
    print(f"✅ Fundamentals saved. Sentiment: {state['sentiment_label']}")

if __name__ == "__main__":
    fetch_fundamentals()
