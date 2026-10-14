import ccxt
import pandas as pd
import os
import json
from datetime import datetime, timezone

# ==========================================
# CONFIG
# ==========================================
JOURNAL_FILE = 'trade_journal.csv'
STATS_FILE = 'bot_stats.json'

TAKER_FEE = 0.0005       # 0.05% per side (adjust to your OKX tier)
SLIPPAGE  = 0.0005       # 0.05% per side
TIME_STOP_DAYS = 7

exchange = ccxt.okx({
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

def atomic_write(path, content):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: f.write(content)
    os.replace(tmp, path)

def fetch_range(symbol, since_ms, timeframe='5m', max_bars=1000):
    all_candles = []
    cursor = since_ms
    for _ in range(20):
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=max_bars)
        if not batch: break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        if len(batch) < max_bars or last_ts >= exchange.milliseconds(): break
        cursor = last_ts + 1
    if not all_candles: return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    # Drop currently-forming candle
    now_ms = exchange.milliseconds()
    df = df[df['timestamp'].astype('int64') // 10**6 < now_ms - 60000]
    return df.drop_duplicates(subset='timestamp').reset_index(drop=True)

def r_multiple(direction, entry, sl, exit_price):
    risk = abs(entry - sl)
    if risk == 0: return 0.0
    return (exit_price - entry)/risk if direction == 'LONG' else (entry - exit_price)/risk

def resolve_trade(row, candles):
    entry_time = pd.to_datetime(row['Date'], utc=True)
    time_stop_ts = entry_time + pd.Timedelta(days=TIME_STOP_DAYS)
    direction = row['Direction']
    entry = float(row['Entry'])
    sl    = float(row['Stop_Loss'])
    tp    = float(row['Take_Profit'])

    future = candles[candles['timestamp'] > entry_time].reset_index(drop=True)
    if future.empty: return (None, None, None)

    for _, c in future.iterrows():
        ts = c['timestamp']
        # Time stop check (exit at this candle's open)
        if ts > time_stop_ts:
            exit_price = float(c['open'])
            return ('TIME_STOP', round(r_multiple(direction, entry, sl, exit_price), 3),
                    ts.strftime("%Y-%m-%d %H:%M:%S"))

        high, low = float(c['high']), float(c['low'])
        if direction == 'LONG':
            hit_sl = low <= sl; hit_tp = high >= tp
        else:
            hit_sl = high >= sl; hit_tp = low <= tp

        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        if hit_sl and hit_tp:
            # ⭐ Pessimistic default: same-candle = LOSS
            return ('LOSS', -1.0, ts_str)
        elif hit_sl:
            return ('LOSS', -1.0, ts_str)
        elif hit_tp:
            return ('WIN', 2.0, ts_str)

    return (None, None, None)

def apply_costs(r_mult, entry, sl, position_size):
    """Subtract fees+slippage from R."""
    risk_dist = abs(entry - sl)
    if risk_dist == 0 or position_size == 0: return r_mult
    notional = position_size * entry
    fee_cost = notional * TAKER_FEE * 2
    slip_cost = notional * SLIPPAGE * 2
    total_cost = fee_cost + slip_cost
    risk_dollars = position_size * risk_dist
    cost_r = total_cost / risk_dollars
    return r_mult - cost_r

def main():
    if not os.path.isfile(JOURNAL_FILE):
        print("No journal."); return
    df = pd.read_csv(JOURNAL_FILE)
    if df.empty: print("Journal empty."); return

    for col in ['Signal_ID','Strategy_Version','R_Multiple','Close_Date']:
        if col not in df.columns: df[col] = None

    updated = False
    for idx, row in df.iterrows():
        if row['Status'] != 'PENDING': continue
        symbol = row['Symbol']
        try:
            entry_time = pd.to_datetime(row['Date'], utc=True)
            since_ms = int(entry_time.timestamp() * 1000) - 60000
            candles = fetch_range(symbol, since_ms)
            if candles.empty:
                print(f"{symbol}: no candles yet."); continue

            status, r_mult, close_date = resolve_trade(row, candles)
            if status is None:
                print(f"{row['Date']} {symbol}: still open."); continue

            # Apply costs
            r_mult = apply_costs(r_mult,
                                 float(row['Entry']),
                                 float(row['Stop_Loss']),
                                 float(row['Position_Size']))

            df.at[idx, 'Status'] = status
            df.at[idx, 'R_Multiple'] = round(r_mult, 3)
            df.at[idx, 'Close_Date'] = close_date
            updated = True
            print(f"✅ {row['Date']} {symbol} {row['Direction']} -> {status} ({r_mult:+.2f}R)")
        except Exception as e:
            print(f"Resolve err {symbol}: {e}")

    if updated:
        atomic_write(JOURNAL_FILE, df.to_csv(index=False))
        print("Journal updated.")

    # -------- Stats --------
    closed = df[df['Status'].isin(['WIN','LOSS','TIME_STOP'])].copy()
    closed['R_Multiple'] = pd.to_numeric(closed['R_Multiple'], errors='coerce').fillna(0)

    wins       = int((df['Status'] == 'WIN').sum())
    losses     = int((df['Status'] == 'LOSS').sum())
    time_stops = int((df['Status'] == 'TIME_STOP').sum())
    pending    = int((df['Status'] == 'PENDING').sum())
    total_res  = wins + losses + time_stops

    win_rate = (wins / total_res * 100) if total_res else 0.0
    total_r  = float(closed['R_Multiple'].sum()) if not closed.empty else 0.0
    avg_r    = (total_r / total_res) if total_res else 0.0

    print(f"\n--- STATS ---")
    print(f"Total: {len(df)} | Wins: {wins} | Losses: {losses} | TimeStops: {time_stops} | Pending: {pending}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total R: {total_r:+.2f}R | Avg R/trade: {avg_r:+.3f}R")

    stats = {
        "total_trades": int(len(df)),
        "closed_trades": total_res,
        "wins": wins, "losses": losses, "time_stops": time_stops, "pending": pending,
        "win_rate": round(win_rate, 2),
        "total_r": round(total_r, 3),
        "avg_r_per_trade": round(avg_r, 4),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    atomic_write(STATS_FILE, json.dumps(stats, indent=4))
    print("Stats saved.")

if __name__ == "__main__":
    main()
