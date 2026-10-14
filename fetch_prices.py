import ccxt
import pandas as pd
import time
import csv
import os
import requests
import json
from datetime import datetime, timezone

# ==========================================
# CONFIG
# ==========================================
STRATEGY_VERSION = "swing-v3.0-phase1"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HEALTHCHECK_URL = os.getenv("HEALTHCHECK_URL")  # optional

JOURNAL_FILE = 'trade_journal.csv'
COOLDOWN_FILE = 'cooldown_tracker.json'

ACCOUNT_BALANCE = 10000.0
RISK_PERCENT = 0.01
COOLDOWN_SECONDS = 3600

SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
           'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT']

exchange = ccxt.okx({
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# ==========================================
# HELPERS
# ==========================================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets missing."); return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

def ping_healthcheck():
    if not HEALTHCHECK_URL: return
    try: requests.get(HEALTHCHECK_URL, timeout=5)
    except Exception as e: print(f"Healthcheck failed: {e}")

def atomic_write(path, content):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: f.write(content)
    os.replace(tmp, path)

def load_cooldowns():
    if os.path.isfile(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE, 'r') as f:
                data = json.load(f)
            return {k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S") for k, v in data.items()}
        except Exception as e: print(f"Cooldown read err: {e}")
    return {}

def save_cooldowns(cooldowns):
    serial = {k: v.strftime("%Y-%m-%d %H:%M:%S") for k, v in cooldowns.items()}
    atomic_write(COOLDOWN_FILE, json.dumps(serial, indent=4))

def get_existing_signal_ids():
    if not os.path.isfile(JOURNAL_FILE): return set()
    try:
        df = pd.read_csv(JOURNAL_FILE)
        if 'Signal_ID' in df.columns:
            return set(df['Signal_ID'].dropna().tolist())
    except Exception as e: print(f"Journal read err: {e}")
    return set()

def append_journal(row):
    file_exists = os.path.isfile(JOURNAL_FILE)
    with open(JOURNAL_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Signal_ID','Strategy_Version','Date','Symbol','Direction',
                             'Entry','Stop_Loss','Take_Profit','Position_Size',
                             'Risk_Amount','Status','R_Multiple','Close_Date'])
        writer.writerow(row)

# ==========================================
# CLOSED-CANDLE ANALYSIS (the critical fix)
# ==========================================
def analyze_timeframe(symbol, timeframe, limit=100, retries=3):
    """Returns the last CLOSED candle with indicators. Drops the forming candle."""
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 55:
                raise ValueError(f"Insufficient candles: {len(ohlcv) if ohlcv else 0}")

            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            # ⭐ THE FIX: drop the currently-forming candle
            df = df.iloc[:-1].reset_index(drop=True)

            if df[['open','high','low','close']].isnull().any().any():
                raise ValueError("NaN in OHLC")

            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['Recent_High'] = df['high'].rolling(window=20).max()
            df['Recent_Low'] = df['low'].rolling(window=20).min()

            last = df.iloc[-1]
            if pd.isna(last['RSI']) or pd.isna(last['EMA_50']):
                raise ValueError("Indicator NaN")
            return last
        except Exception as e:
            if attempt == retries - 1: raise e
            time.sleep(2)

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"Strategy: {STRATEGY_VERSION}")
    print(f"Scan (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    cooldowns = load_cooldowns()
    existing_ids = get_existing_signal_ids()
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for symbol in SYMBOLS:
        try:
            print(f"--- {symbol} ---")
            daily  = analyze_timeframe(symbol, '1d')
            four_h = analyze_timeframe(symbol, '4h')
            one_h  = analyze_timeframe(symbol, '1h')

            daily_bias  = "LONG" if daily['EMA_20']  > daily['EMA_50']  else "SHORT"
            four_h_bias = "LONG" if four_h['EMA_20'] > four_h['EMA_50'] else "SHORT"

            signal = "WAIT"; reason = "No setup."

            if daily_bias == "LONG" and four_h_bias == "SHORT":
                if four_h['RSI'] < 80 and one_h['RSI'] > 40:
                    signal = "LONG"; reason = "Daily Bullish. 4H Oversold. 1H Recovering."
                else:
                    reason = f"Daily Bullish. 4H pullback not ready (RSI={four_h['RSI']:.1f})."
            elif daily_bias == "SHORT" and four_h_bias == "LONG":
                if four_h['RSI'] > 65 and one_h['RSI'] < 60:
                    signal = "SHORT"; reason = "Daily Bearish. 4H Overbought. 1H Rolling Over."
                else:
                    reason = f"Daily Bearish. 4H rally not ready (RSI={four_h['RSI']:.1f})."
            elif daily_bias == "LONG" and four_h_bias == "LONG":
                if one_h['RSI'] < 45:
                    signal = "LONG"; reason = "Daily/4H Bullish. 1H Pullback."
                else:
                    reason = "Daily/4H Bullish. Waiting 1H pullback (RSI<45)."
            elif daily_bias == "SHORT" and four_h_bias == "SHORT":
                if one_h['RSI'] > 55:
                    signal = "SHORT"; reason = "Daily/4H Bearish. 1H Rally."
                else:
                    reason = "Daily/4H Bearish. Waiting 1H rally (RSI>55)."

            print(f"  Daily={daily_bias} 4H={four_h_bias} | 4H RSI={four_h['RSI']:.1f} 1H RSI={one_h['RSI']:.1f}")
            print(f"  Signal: {signal} | {reason}")

            if signal == "WAIT": continue

            # Cooldown
            last_signal = cooldowns.get(symbol)
            if last_signal and (now - last_signal).total_seconds() < COOLDOWN_SECONDS:
                print(f"  Cooldown active. Skip."); continue

            # ⭐ Idempotency: signal_id built from symbol+side+1H-candle-close+version
            candle_close = one_h['timestamp'].strftime("%Y%m%d%H%M")
            signal_id = f"{symbol.replace('/','_')}_{signal}_{candle_close}_{STRATEGY_VERSION}"

            if signal_id in existing_ids:
                print(f"  Already logged ({signal_id}). Skip."); continue

            # Risk calc
            entry_price = one_h['close']
            if signal == "LONG":
                stop_loss = one_h['Recent_Low']
                risk_per_unit = entry_price - stop_loss
                take_profit = entry_price + (risk_per_unit * 2)
            else:
                stop_loss = one_h['Recent_High']
                risk_per_unit = stop_loss - entry_price
                take_profit = entry_price - (risk_per_unit * 2)

            if risk_per_unit <= 0:
                print("  Invalid risk distance. Skip."); continue

            risk_amount = ACCOUNT_BALANCE * RISK_PERCENT
            position_size = risk_amount / risk_per_unit
            signal_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

            # ⭐ Log BEFORE alert (prevents duplicate alerts on retries)
            append_journal([
                signal_id, STRATEGY_VERSION, signal_time_str, symbol, signal,
                f"{entry_price:.4f}", f"{stop_loss:.4f}", f"{take_profit:.4f}",
                f"{position_size:.6f}", f"{risk_amount:.2f}",
                "PENDING", "", ""
            ])
            existing_ids.add(signal_id)

            emoji = "📈" if signal == "LONG" else "📉"
            message = (
                f"🚨 *{STRATEGY_VERSION}* 🚨\n\n"
                f"{emoji} *{signal}* {symbol}\n\n"
                f"Entry: `${entry_price:.4f}`\n"
                f"Stop: `${stop_loss:.4f}`\n"
                f"Target: `${take_profit:.4f}`\n\n"
                f"Size: `{position_size:.6f}`\n"
                f"Risk: `${risk_amount:.2f}`\n"
                f"R:R: `1:2`\n\n"
                f"🧠 {reason}\n"
                f"🕐 Signal: {signal_time_str} UTC"
            )
            send_telegram(message)
            print(f"  ✅ Alert sent.")
            cooldowns[symbol] = now

        except Exception as e:
            print(f"  ❌ Error on {symbol}: {e}")
            continue
        time.sleep(2)

    save_cooldowns(cooldowns)
    ping_healthcheck()
    print("Scan complete.")

if __name__ == "__main__":
    main()
