import ccxt
import pandas as pd
import time
import csv
import os
import requests
import json
from datetime import datetime

# ==========================================
# TELEGRAM SETTINGS
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================================
# CHIEF AGENT (DYNAMIC RISK SETTINGS)
# ==========================================
STATS_FILE = 'bot_stats.json'
COOLDOWN_FILE = 'cooldown_tracker.json'
JOURNAL_FILE = 'trade_journal.csv'

# Defaults
ACCOUNT_BALANCE = 10000
RISK_PERCENT = 0.01
WIN_RATE = 50.0

# Read Stats and Adjust Risk
if os.path.isfile(STATS_FILE):
    try:
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
            WIN_RATE = stats.get('win_rate', 50.0)
            total_trades = stats.get('total_trades', 0)
            ACCOUNT_BALANCE = stats.get('current_balance', 10000)
            
            print(f"🧠 Memory Loaded: Win Rate = {WIN_RATE}% over {total_trades} trades. Balance: ${ACCOUNT_BALANCE:.2f}")
            
            if total_trades > 5:
                if WIN_RATE >= 60:
                    RISK_PERCENT = 0.015 # Increase risk to 1.5% if winning
                    print("📈 Performance strong. Adjusting risk UP to 1.5%.")
                elif WIN_RATE < 40:
                    RISK_PERCENT = 0.005 # Decrease risk to 0.5% if losing
                    print("📉 Performance poor. Adjusting risk DOWN to 0.5%.")
    except Exception as e:
        print(f"Error reading stats: {e}")

# Load Cooldowns
last_signal_times = {}
if os.path.isfile(COOLDOWN_FILE):
    try:
        with open(COOLDOWN_FILE, 'r') as f:
            cooldown_data = json.load(f)
            for symbol, last_time in cooldown_data.items():
                last_signal_times[symbol] = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error reading cooldowns: {e}")

# ==========================================
# SYMBOLS & EXCHANGE
# ==========================================
SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
    'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT'
]

exchange = ccxt.okx({
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# ==========================================
# TELEGRAM FUNCTION
# ==========================================
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ==========================================
# HELPER FUNCTION
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
            if i == retries - 1: raise e
            time.sleep(2)

# ==========================================
# MAIN EXECUTION
# ==========================================
print(f"Chief Agent: Scanning 6 Assets...")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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

        print(f"   Daily Bias: {daily_bias} | 4H Bias: {four_h_bias}")
        print(f"   *** SIGNAL: {signal} *** | Reason: {reason}")

        if signal != "WAIT":
            current_time = datetime.now()
            last_time = last_signal_times.get(symbol)
            
            # Cooldown Check (1 hour)
            if last_time is None or (current_time - last_time).total_seconds() > 3600:
                entry_price = five_m['close']
                
                if signal == "LONG":
                    stop_loss = one_h['Recent_Low']
                    risk_per_btc = entry_price - stop_loss
                    take_profit = entry_price + (risk_per_btc * 2)
                else:
                    stop_loss = one_h['Recent_High']
                    risk_per_btc = stop_loss - entry_price
                    take_profit = entry_price - (risk_per_btc * 2)
                
                position_size = (ACCOUNT_BALANCE * RISK_PERCENT) / risk_per_btc
                
                message = f"🚨 *MULTI-TIMEFRAME ALERT* 🚨\n\n{'📈 *LONG*' if signal == 'LONG' else '📉 *SHORT*'} on {symbol}\n\nEntry: ${entry_price:.2f}\nStop Loss: ${stop_loss:.2f}\nTake Profit: ${take_profit:.2f}\n\n⚖️ Risk-to-Reward: 1:2\n📏 Position Size: {position_size:.4f} BTC\n\n🧠 *Reason:* {reason}"
                send_telegram_message(message)
                print(f"   ✅ Signal sent to Telegram for {symbol}!")
                
                # Update Cooldown
                last_signal_times[symbol] = current_time

                # Save to Journal
                file_exists = os.path.isfile(JOURNAL_FILE)
                with open(JOURNAL_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(['Date', 'Symbol', 'Direction', 'Entry', 'Stop_Loss', 'Take_Profit', 'Position_Size', 'Status'])
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, signal, f"{entry_price:.2f}", f"{stop_loss:.2f}", f"{take_profit:.2f}", f"{position_size:.4f}", "PENDING"])
            else:
                print(f"   ⏳ Skipping alert for {symbol} (1-hour cooldown active).")

        time.sleep(2)

    except Exception as e:
        print(f"   ❌ Error analyzing {symbol}: {e}")
        time.sleep(2)

# Save Cooldowns back to file
with open(COOLDOWN_FILE, 'w') as f:
    cooldown_serializable = {k: v.strftime("%Y-%m-%d %H:%M:%S") for k, v in last_signal_times.items()}
    json.dump(cooldown_serializable, f, indent=4)

print("Scan complete. Cooldowns saved.")
