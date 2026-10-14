import requests
import json
import os
from datetime import datetime, timezone, timedelta

FUNDAMENTAL_FILE = 'fundamental_state.json'
VETO_FILE = 'veto_state.json'
VETO_DURATION_HOURS = 6

# ==========================================
# CONTEXT-AWARE SENTIMENT PATTERNS
# ==========================================
# Format: (subject, verbs_positive, verbs_negative)
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
        "negative": [],  # Hack itself is the negative
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
        "negative": [],  # Ban itself is negative
        "magnitude": 0.8
    },
]

# ==========================================
# PER-ASSET RELEVANCE MAP
# ==========================================
ASSET_KEYWORDS = {
    "BTC": ["BITCOIN", "BTC", "SATOSHI"],
    "ETH": ["ETHEREUM", "ETH", "VITALIK", "ETHER"],
    "SOL": ["SOLANA", "SOL"],
    "BNB": ["BNB", "BINANCE", "CZ"],
    "XRP": ["XRP", "RIPPLE"],
    "DOGE": ["DOGECOIN", "DOGE", "ELON"]
}

# Global-impact subjects (affect ALL assets if detected)
GLOBAL_SUBJECTS = ["FED", "FEDERAL RESERVE", "SEC", "CRYPTO MARKET", "EXCHANGE HACK", "BINANCE HACK", "COINBASE HACK"]


# ==========================================
# SCORING ENGINE
# ==========================================
def score_headline(title):
    """Returns (score, reason). Score ranges -1.0 to +1.0."""
    upper = title.upper()
    best_score = 0.0
    best_reason = None

    for pattern in SENTIMENT_PATTERNS:
        subject_found = any(s in upper for s in pattern["subjects"])
        if not subject_found:
            continue

        # Check for positive verbs
        for verb in pattern["positive"]:
            if verb in upper:
                score = pattern["magnitude"]
                if score > best_score:
                    best_score = score
                    best_reason = f"{pattern['subjects'][0]} + {verb} (bullish)"
                break
        # Check for negative verbs
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
    """Returns a list of affected asset symbols. Empty = global impact."""
    upper = title.upper()
    affected = []
    for asset, keywords in ASSET_KEYWORDS.items():
        if any(kw in upper for kw in keywords):
            affected.append(asset)

    # If global subject detected and no specific asset, applies to all
    if not affected and any(g in upper for g in GLOBAL_SUBJECTS):
        return ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
    return affected


# ==========================================
# MAIN FETCH
# ==========================================
def fetch_fundamentals():
    print("📰 Phase 4 News Agent starting...")

    state = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "api_status": "OK",
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral",
        "x_sentiment": 0.0,
        "headlines_scored": [],
        "per_asset_vetoes": {}  # {asset: {"reason": "...", "expires": "..."}}
    }

    try:
        # 1. Overall market sentiment
        try:
            r1 = requests.get("https://cryptocurrency.cv/api/sentiment", timeout=10)
            if r1.status_code == 200:
                data = r1.json()
                state["sentiment_score"] = float(data.get("score", 0.0))
                state["sentiment_label"] = data.get("label", "Neutral")
        except Exception as e:
            print(f"  ⚠️ Sentiment endpoint failed: {e}")

        # 2. X/Twitter sentiment
        try:
            r2 = requests.get("https://cryptocurrency.cv/api/social/x/sentiment", timeout=10)
            if r2.status_code == 200:
                data = r2.json()
                state["x_sentiment"] = float(data.get("score", 0.0))
        except Exception as e:
            print(f"  ⚠️ X sentiment endpoint failed: {e}")

        # 3. Breaking news with context-aware scoring
        try:
            r3 = requests.get("https://cryptocurrency.cv/api/breaking", timeout=10)
            if r3.status_code == 200:
                data = r3.json()
                articles = data.get("articles", [])[:10]

                for art in articles:
                    title = art.get("title", "")
                    if not title:
                        continue
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

                    # Trigger a veto if score is very negative
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

    # Write output atomically
    tmp = FUNDAMENTAL_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=4)
    os.replace(tmp, FUNDAMENTAL_FILE)

    print(f"✅ Fundamentals saved. API: {state['api_status']}, Sentiment: {state['sentiment_label']}, "
          f"Active vetoes: {len(state['per_asset_vetoes'])}")
    return state


def is_vetoed(asset, state):
    """Check if an asset is currently vetoed. Returns (bool, reason)."""
    if not state: return False, None
    vetoes = state.get("per_asset_vetoes", {})
    if asset not in vetoes: return False, None
    veto = vetoes[asset]
    expires = datetime.strptime(veto["expires"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires:
        return False, None
    return True, veto["reason"]


def risk_multiplier(state):
    """Returns a multiplier (0.0 to 1.0) for position size based on news sentiment."""
    if not state: return 1.0
    score = state.get("sentiment_score", 0.0)
    if score < -0.6: return 0.0   # Hard veto globally
    if score < -0.3: return 0.5   # Half risk
    return 1.0


if __name__ == "__main__":
    fetch_fundamentals()
