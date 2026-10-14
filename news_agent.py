import requests
import json
import os
from datetime import datetime, timezone, timedelta

FUNDAMENTAL_FILE = 'fundamental_state.json'
VETO_DURATION_HOURS = 6

SENTIMENT_PATTERNS = [
    {
        "subjects": ["SEC", "SECURITIES AND EXCHANGE"],
        "positive": ["APPROVE", "APPROVES", "SETTLE", "SETTLES", "CLEAR", "CLEARS", "GREENLIGHT"],
        "negative": ["SUE", "SUES", "CHARGES", "INVESTIGATE", "INVESTIGATES", "BAN", "BANS", "REJECT", "REJECTS"],
        "magnitude": 0.8
    },
    {
        "subjects": ["FED", "FEDERAL RESERVE", "FOMC"],
        "positive": ["CUT", "CUTS", "PIVOT", "DOVISH"],
        "negative": ["HIKE", "HIKES", "RAISE", "RAISES", "HAWKISH"],
        "magnitude": 0.6
    },
    {
        "subjects": ["HACK", "HACKED", "EXPLOIT", "EXPLOITED", "STOLEN", "DRAINED", "BREACH"],
        "positive": ["RECOVER", "RECOVERED", "REFUNDED", "RESTORED"],
        "negative": [],
        "magnitude": 0.9
    },
    {
        "subjects": ["ETF"],
        "positive": ["APPROVE", "APPROVES", "LAUNCH", "LAUNCHES", "FILED"],
        "negative": ["REJECT", "REJECTS", "DELAY", "DELAYS"],
        "magnitude": 0.7
    },
    {
        "subjects": ["BAN", "BANNED", "PROHIBIT", "PROHIBITED"],
        "positive": [],
        "negative": [],
        "magnitude": 0.8
    },
]

ASSET_KEYWORDS = {
    "BTC": ["BITCOIN", "BTC", "SATOSHI"],
    "ETH": ["ETHEREUM", "ETH", "VITALIK", "ETHER"],
    "SOL": ["SOLANA", "SOL"],
    "BNB": ["BNB", "BINANCE", "CZ"],
    "XRP": ["XRP", "RIPPLE"],
    "DOGE": ["DOGECOIN", "DOGE", "ELON"]
}

GLOBAL_SUBJECTS = ["FED", "FEDERAL RESERVE", "SEC", "CRYPTO MARKET", "EXCHANGE HACK", "BINANCE HACK", "COINBASE HACK"]

def score_headline(title):
    upper = title.upper()
    best_score = 0.0
    best_reason = None

    for pattern in SENTIMENT_PATTERNS:
        subject_found = any(s in upper for s in pattern["subjects"])
        if not subject_found:
            continue

        for verb in pattern["positive"]:
            if verb in upper:
                score = pattern["magnitude"]
                if score > best_score:
                    best_score = score
                    best_reason = f"{pattern['subjects'][0]} + {verb} (bullish)"
                break
        for verb in pattern["negative"]:
            if verb in upper:
                score = -pattern["magnitude"]
                if score < best_score or best_score == 0.0:
                    if score < best_score or best_score >= 0.0:
                        best_score = score
                        best_reason = f"{pattern['subjects'][0]} + {verb} (bearish)"
                break

    return best_score, best_reason

def affected_assets(title):
    upper = title.upper()
    affected = []
    for asset, keywords in ASSET_KEYWORDS.items():
        if any(kw in upper for kw in keywords):
            affected.append(asset)

    if not affected and any(g in upper for g in GLOBAL_SUBJECTS):
        return ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
    return affected

def fetch_fundamentals():
    print("📰 Phase 4 News Agent starting...")
    state = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "api_status": "OK",
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral",
        "x_sentiment": 0.0,
        "headlines_scored": [],
        "per_asset_vetoes": {}
    }

    try:
        try:
            r1 = requests.get("https://cryptocurrency.cv/api/sentiment", timeout=10)
            if r1.status_code == 200:
                data = r1.json()
                state["sentiment_score"] = float(data.get("score", 0.0))
                state["sentiment_label"] = data.get("label", "Neutral")
        except Exception as e:
            print(f"  ⚠️ Sentiment endpoint failed: {e}")
            state["api_status"] = "DEGRADED"

        try:
            r2 = requests.get("https://cryptocurrency.cv/api/social/x/sentiment", timeout=10)
            if r2.status_code == 200:
                data = r2.json()
                state["x_sentiment"] = float(data.get("score", 0.0))
        except Exception as e:
            print(f"  ⚠️ X sentiment endpoint failed: {e}")

        try:
            r3 = requests.get("https://cryptocurrency.cv/api/breaking", timeout=10)
            if r3.status_code == 200:
                data = r3.json()
                articles = data.get("articles", [])[:10]
                for art in articles:
                    title = art.get("title", "")
                    if not title: continue
                    score, reason = score_headline(title)
                    affected = affected_assets(title)
                    entry = {
                        "title": title,
                        "score": round(score, 2),
                        "reason": reason or "no pattern match",
                        "affected_assets": affected or ["GLOBAL"],
                        "url": art.get("url", "")
                    }
                    state["headlines_scored"].append(entry)

                    if score <= -0.6:
                        expires = (datetime.now(timezone.utc) + timedelta(hours=VETO_DURATION_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
                        targets = affected if affected else ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
                        for asset in targets:
                            state["per_asset_vetoes"][asset] = {
                                "reason": f"{reason}: {title[:80]}",
                                "expires": expires
                            }
        except Exception as e:
            print(f"  ⚠️ News endpoint failed: {e}")

    except Exception as e:
        print(f"❌ Fatal news agent error: {e}")
        state["api_status"] = "FAILED"

    tmp = FUNDAMENTAL_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=4)
    os.replace(tmp, FUNDAMENTAL_FILE)

    print(f"✅ Fundamentals saved. API: {state['api_status']}, Sentiment: {state['sentiment_label']}, Active vetoes: {len(state['per_asset_vetoes'])}")
    return state

def is_vetoed(asset, state):
    if not state: return False, None
    vetoes = state.get("per_asset_vetoes", {})
    if asset not in vetoes: return False, None
    veto = vetoes[asset]
    expires = datetime.strptime(veto["expires"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires:
        return False, None
    return True, veto["reason"]

def risk_multiplier(state):
    if not state: return 1.0
    score = state.get("sentiment_score", 0.0)
    if score < -0.6: return 0.0
    if score < -0.3: return 0.5
    return 1.0

if __name__ == "__main__":
    fetch_fundamentals()
