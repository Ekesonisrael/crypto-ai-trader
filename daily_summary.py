import ccxt
import pandas as pd
import os
import requests
from datetime import datetime
import news_agent

# TELEGRAM SETTINGS
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
    'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT'
]

exchange = ccxt.okx({
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

def get_bias(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    last = df.iloc[-1]
    return "LONG" if last['EMA_20'] > last['EMA_50'] else "SHORT"

print("Generating Market Pulse...")

# Fetch Fundamental Data Once
news_status, news_reason = news_agent.check_news_impact()

message = "📊 *MARKET PULSE (4-HOUR DIGEST)* 📊\n"
message += f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
message += f"📰 *Fundamental News:* `{news_status}`\n"
message += f"_{news_reason}_\n\n"
message += "Here is what your AI is seeing right now:\n\n"

for symbol in SYMBOLS:
    try:
        daily_bias = get_bias(symbol, '1d')
        four_h_bias = get_bias(symbol, '4h')
        
        # Get Funding Rate (Sentiment)
        funding_data = exchange.fetch_funding_rate(symbol)
        funding_rate = funding_data['fundingRate'] * 100
        
        coin = symbol.split('/')[0]
        message += f"🪙 *{coin}*\n"
        message += f"Daily: `{daily_bias}` | 4H: `{four_h_bias}`\n"
        message += f"Funding Rate: `{funding_rate:.4f}%`\n\n"
    except Exception as e:
        message += f"🪙 *{symbol.split('/')[0]}* - Error: {e}\n\n"

message += "🤖 _The bot is alive and waiting for a perfect setup._"

send_telegram_message(message)
print("Market Pulse sent to Telegram!")
