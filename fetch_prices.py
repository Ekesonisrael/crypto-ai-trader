import ccxt
import pandas as pd
import time
import csv
import os
import requests
from datetime import datetime

# ==========================================
# TELEGRAM SETTINGS
# ==========================================
import os
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# ==========================================
# CHIEF AGENT (YOUR ACCOUNT SETTINGS)
# ==========================================
ACCOUNT_BALANCE = 10000
RISK_PERCENT = 0.01
SYMBOL = 'BTC/USDT:USDT'
TIMEFRAME = '5m'
LIMIT = 100
JOURNAL_FILE = 'trade_journal.csv'

# Setup Bybit Futures
bybit = ccxt.bybit({
    'timeout': 30000,
    'options': {
        'defaultType': 'swap',
        'fetchMarkets': ['linear'],
    }
})

# ==========================================
# TELEGRAM FUNCTION
# ==========================================
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

print("Starting the AI Trading Agent. Press Ctrl+C to stop.\n")

# ==========================================
# THE AUTONOMOUS LOOP
# ==========================================
while True:
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Chief Agent: Scanning {SYMBOL} on {TIMEFRAME} timeframe...")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. Fetch the OHLCV data
        ohlcv = bybit.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 2. Calculate Indicators
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['Recent_High'] = df['high'].rolling(window=20).max()
        df['Recent_Low'] = df['low'].rolling(window=20).min()

        last_candle = df.iloc[-1]
        entry_price = last_candle['close']

        # 3. Trading Logic (Charts Agent) - STRICT RULES RESTORED
        signal = "WAIT"
        if last_candle['EMA_20'] > last_candle['EMA_50'] and last_candle['RSI'] > 50:
            signal = "LONG"
        elif last_candle['EMA_20'] < last_candle['EMA_50'] and last_candle['RSI'] < 50:
            signal = "SHORT"

        # 4. Sentiment Agent
        try:
            funding_data = bybit.fetch_funding_rate(SYMBOL)
            funding_rate = funding_data['fundingRate']
            funding_percent = funding_rate * 100
        except:
            funding_rate = 0

        print("--- CHARTS AGENT ---")
        print(f"Close Price: ${entry_price:.2f} | RSI: {last_candle['RSI']:.2f}")
        print(f"EMA_20: ${last_candle['EMA_20']:.2f} | EMA_50: ${last_candle['EMA_50']:.2f}")
        print(f"*** SIGNAL: {signal} ***\n")

        # 5. Risk Management & Execution
        if signal == "SHORT":
            stop_loss = last_candle['Recent_High']
            risk_per_btc = stop_loss - entry_price
            take_profit = entry_price - (risk_per_btc * 2)
            position_size = (ACCOUNT_BALANCE * RISK_PERCENT) / risk_per_btc
            
            print(f"--- SKEPTIC AGENT ---\nDirection: SHORT\nEntry: ${entry_price:.2f}\nStop Loss: ${stop_loss:.2f}\nTake Profit: ${take_profit:.2f}")
            print(f"\n--- CHIEF AGENT ---\nPosition Size: {position_size:.4f} BTC\nSTATUS: APPROVED ✅")
            
            # SEND TO TELEGRAM
            message = f"🚨 *NEW TRADE ALERT* 🚨\n\n📉 *SHORT* on {SYMBOL}\n\nEntry: ${entry_price:.2f}\nStop Loss: ${stop_loss:.2f}\nTake Profit: ${take_profit:.2f}\n\n⚖️ Risk-to-Reward: 1:2\n📏 Position Size: {position_size:.4f} BTC"
            send_telegram_message(message)
            print("Signal sent to Telegram!")

            # SAVE TO JOURNAL
            file_exists = os.path.isfile(JOURNAL_FILE)
            with open(JOURNAL_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Date', 'Symbol', 'Direction', 'Entry', 'Stop_Loss', 'Take_Profit', 'Position_Size', 'Status'])
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), SYMBOL, "SHORT", f"{entry_price:.2f}", f"{stop_loss:.2f}", f"{take_profit:.2f}", f"{position_size:.4f}", "PENDING"])

        elif signal == "LONG":
            stop_loss = last_candle['Recent_Low']
            risk_per_btc = entry_price - stop_loss
            take_profit = entry_price + (risk_per_btc * 2)
            position_size = (ACCOUNT_BALANCE * RISK_PERCENT) / risk_per_btc
            
            print(f"--- SKEPTIC AGENT ---\nDirection: LONG\nEntry: ${entry_price:.2f}\nStop Loss: ${stop_loss:.2f}\nTake Profit: ${take_profit:.2f}")
            print(f"\n--- CHIEF AGENT ---\nPosition Size: {position_size:.4f} BTC\nSTATUS: APPROVED ✅")
            
            # SEND TO TELEGRAM
            message = f"🚨 *NEW TRADE ALERT* 🚨\n\n📈 *LONG* on {SYMBOL}\n\nEntry: ${entry_price:.2f}\nStop Loss: ${stop_loss:.2f}\nTake Profit: ${take_profit:.2f}\n\n⚖️ Risk-to-Reward: 1:2\n📏 Position Size: {position_size:.4f} BTC"
            send_telegram_message(message)
            print("Signal sent to Telegram!")

            # SAVE TO JOURNAL
            file_exists = os.path.isfile(JOURNAL_FILE)
            with open(JOURNAL_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Date', 'Symbol', 'Direction', 'Entry', 'Stop_Loss', 'Take_Profit', 'Position_Size', 'Status'])
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), SYMBOL, "LONG", f"{entry_price:.2f}", f"{stop_loss:.2f}", f"{take_profit:.2f}", f"{position_size:.4f}", "PENDING"])

        else:
            print("--- CHIEF AGENT ---\nSTATUS: NO TRADE ❌ (Waiting for a better setup)")

        print("\nWaiting 60 seconds for the next scan...")
        time.sleep(60)

    except KeyboardInterrupt:
        print("\nBot stopped manually. Goodbye!")
        break
    except Exception as e:
        print(f"\nNetwork or API Error: {e}")
        print("Retrying in 30 seconds...")
        time.sleep(30)