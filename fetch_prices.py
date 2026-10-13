import ccxt
import pandas as pd
import time
import csv
import os
import requests
from datetime import datetime

# ==========================================
# TELEGRAM SETTINGS (Loaded from GitHub Secrets)
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================================
# CHIEF AGENT (YOUR ACCOUNT SETTINGS)
# ==========================================
ACCOUNT_BALANCE = 10000
RISK_PERCENT = 0.01
JOURNAL_FILE = 'trade_journal.csv'

# THE 6 ASSETS TO SCAN (Binance Futures Symbols)
SYMBOLS = [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    'BNB/USDT',
    'XRP/USDT',
    'DOGE/USDT'
]

# Setup Binance Futures
exchange = ccxt.binance({
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
    }
})

# ==========================================
# TELEGRAM FUNCTION
# ==========================================
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Skipping alert.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ==========================================
# HELPER FUNCTION TO ANALYZE ANY TIMEFRAME
# ==========================================
def analyze_timeframe(symbol, timeframe, limit=100, retries=3):
    for i in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            df['Recent_High'] = df['high'].rolling(window=20).max()
            df['Recent_Low'] = df['low'].rolling(window=20).min()
            
            return df.iloc[-1]
            
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(2)

# ==========================================
# MAIN EXECUTION (Runs once per GitHub Action)
# ==========================================
print(f"Chief Agent: Scanning 6 Assets...")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

last_signal_times = {symbol: None for symbol in SYMBOLS}

for symbol in SYMBOLS:
    try:
        print(f"--- Analyzing {symbol} ---")
        
        daily = analyze_timeframe(symbol, '1d')
        four_h = analyze_timeframe(symbol, '4h')
        one_h = analyze_timeframe(symbol, '1h')
        five_m = analyze_timeframe(symbol, '5m')

        daily_bias = "LONG" if daily['EMA_20'] > daily['EMA_50'] else "SHORT"
        four_h_bias = "LONG" if four_h['EMA_20'] > four_h['EMA_50'] else "SHORT"

        signal = "WAIT"
        reason = "Higher timeframes do not agree."

        if daily_bias == "LONG" and four_h_bias == "LONG":
            if one_h['RSI'] < 45:
                if five_m['RSI'] > 50:
                    signal = "LONG"
                    reason = "Daily/4H Bullish. 1H Pullback. 5m Trigger."
                else:
                    reason = "Daily/4H Bullish. 1H Pullback found. Waiting for 5m trigger."
            else:
                reason = "Daily/4H Bullish. Waiting for 1H pullback (RSI < 45)."

        elif daily_bias == "SHORT" and four_h_bias == "SHORT":
            if one_h['RSI'] > 55:
                if five_m['RSI'] < 50:
                    signal = "SHORT"
                    reason = "Daily/4H Bearish. 1H Rally. 5m Trigger."
                else:
                    reason = "Daily/4H Bearish. 1H Rally found. Waiting for 5m trigger."
            else:
                reason = "Daily/4H Bearish. Waiting for 1H rally (RSI > 55)."

        print(f"   Daily Bias: {daily_bias} (EMA20: ${daily['EMA_20']:.2f} vs EMA50: ${daily['EMA_50']:.2f})")
        print(f"   4H Bias:    {four_h_bias} (EMA20: ${four_h['EMA_20']:.2f} vs EMA50: ${four_h['EMA_50']:.2f})")
        print(f"   1H RSI:     {one_h['RSI']:.2f}")
        print(f"   5m Close:   ${five_m['close']:.2f} | 5m RSI: {five_m['RSI']:.2f}")
        print(f"   *** SIGNAL: {signal} *** | Reason: {reason}")

        if signal == "LONG":
            entry_price = five_m['close']
            stop_loss = one_h['Recent_Low']
            risk_per_btc = entry_price - stop_loss
            take_profit = entry_price + (risk_per_btc * 2)
            position_size = (ACCOUNT_BALANCE * RISK_PERCENT) / risk_per_btc
            
            current_time = datetime.now()
            last_time = last_signal_times[symbol]
            
            if last_time is None or (current_time - last_time).total_seconds() > 3600:
                message = f"🚨 *MULTI-TIMEFRAME ALERT* 🚨\n\n📈 *LONG* on {symbol}\n\nEntry: ${entry_price:.2f}\nStop Loss: ${stop_loss:.2f}\nTake Profit: ${take_profit:.2f}\n\n⚖️ Risk-to-Reward: 1:2\n📏 Position Size: {position_size:.4f} BTC\n\n🧠 *Reason:* {reason}"
                send_telegram_message(message)
                print(f"   ✅ Signal sent to Telegram for {symbol}!")
                last_signal_times[symbol] = current_time

                file_exists = os.path.isfile(JOURNAL_FILE)
                with open(JOURNAL_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(['Date', 'Symbol', 'Direction', 'Entry', 'Stop_Loss', 'Take_Profit', 'Position_Size', 'Status'])
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, "LONG", f"{entry_price:.2f}", f"{stop_loss:.2f}", f"{take_profit:.2f}", f"{position_size:.4f}", "PENDING"])

        elif signal == "SHORT":
            entry_price = five_m['close']
            stop_loss = one_h['Recent_High']
            risk_per_btc = stop_loss - entry_price
            take_profit = entry_price - (risk_per_btc * 2)
            position_size = (ACCOUNT_BALANCE * RISK_PERCENT) / risk_per_btc
            
            current_time = datetime.now()
            last_time = last_signal_times[symbol]
            
            if last_time is None or (current_time - last_time).total_seconds() > 3600:
                message = f"🚨 *MULTI-TIMEFRAME ALERT* 🚨\n\n📉 *SHORT* on {symbol}\n\nEntry: ${entry_price:.2f}\nStop Loss: ${stop_loss:.2f}\nTake Profit: ${take_profit:.2f}\n\n⚖️ Risk-to-Reward: 1:2\n📏 Position Size: {position_size:.4f} BTC\n\n🧠 *Reason:* {reason}"
                send_telegram_message(message)
                print(f"   ✅ Signal sent to Telegram for {symbol}!")
                last_signal_times[symbol] = current_time

                file_exists = os.path.isfile(JOURNAL_FILE)
                with open(JOURNAL_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(['Date', 'Symbol', 'Direction', 'Entry', 'Stop_Loss', 'Take_Profit', 'Position_Size', 'Status'])
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, "SHORT", f"{entry_price:.2f}", f"{stop_loss:.2f}", f"{take_profit:.2f}", f"{position_size:.4f}", "PENDING"])

        time.sleep(2)

    except Exception as e:
        print(f"   ❌ Error analyzing {symbol}: {e}")
        time.sleep(2)

print("Scan complete.")
