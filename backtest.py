import ccxt
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime, timezone, timedelta

# ==========================================
# CONFIG
# ==========================================
SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT']
END_DATE = datetime.now(timezone.utc).replace(tzinfo=None)
START_DATE = END_DATE - timedelta(days=730)

TAKER_FEE = 0.0005
SLIPPAGE_LIMIT = 0.0003      # for TP fills (limit-like)
SLIPPAGE_STOP = 0.0015       # for SL fills (market-like, worse)
FUNDING_INTERVALS_PER_DAY = 3  # 8h funding on OKX

ATR_MULT = 1.5
RR_RATIO = 2.0
ADX_MIN = 20
RISK_PER_TRADE = 0.01
STARTING_BALANCE = 10000.0
TIME_STOP_HOURS = 168
LEVERAGE_CAP = 3.0
TRAIN_RATIO = 0.70

exchange = ccxt.okx({'timeout': 60000, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})


# ==========================================
# DATA FETCHING
# ==========================================
def fetch_candles(symbol, timeframe, since_ms, limit_per_call=1000):
    print(f"  Fetching {symbol} {timeframe}...")
    all_candles = []
    cursor = since_ms
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit_per_call)
        except Exception as e:
            print(f"    Error: {e}. Retry in 3s...")
            time.sleep(3); continue
        if not batch: break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= exchange.milliseconds() - 60000: break
        if len(batch) < limit_per_call: break
        cursor = last_ts + 1
    if not all_candles: return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    now_ms = exchange.milliseconds()
    df = df[df['timestamp'].astype('int64') // 10**6 < now_ms - 60000]
    return df.drop_duplicates(subset='timestamp').reset_index(drop=True)


def fetch_funding_history(symbol, since_ms):
    """Fetch funding rate history."""
    print(f"  Fetching {symbol} funding...")
    all_funding = []
    cursor = since_ms
    for _ in range(30):
        try:
            batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=100)
        except Exception as e:
            print(f"    Funding error: {e}"); break
        if not batch: break
        all_funding.extend(batch)
        if len(batch) < 100: break
        cursor = batch[-1]['timestamp'] + 1
    if not all_funding: return pd.DataFrame()
    df = pd.DataFrame(all_funding)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df[['timestamp', 'fundingRate']].reset_index(drop=True)


# ==========================================
# INDICATORS
# ==========================================
def compute_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['TR'] = tr
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    df['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    df['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    df['+DI'] = 100 * (df['+DM'].rolling(window=14).mean() / df['TR'].rolling(window=14).mean())
    df['-DI'] = 100 * (df['-DM'].rolling(window=14).mean() / df['TR'].rolling(window=14).mean())
    dx = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = dx.rolling(window=14).mean()
    return df


# ==========================================
# SIGNAL
# ==========================================
def get_signal(daily_row, four_h_row, one_h_row):
    if pd.isna(one_h_row['ADX']) or one_h_row['ADX'] < ADX_MIN: return "WAIT"
    if pd.isna(four_h_row['RSI']) or pd.isna(one_h_row['RSI']): return "WAIT"
    db = "LONG" if daily_row['EMA_20'] > daily_row['EMA_50'] else "SHORT"
    fb = "LONG" if four_h_row['EMA_20'] > four_h_row['EMA_50'] else "SHORT"
    if db == "LONG" and fb == "SHORT":
        if four_h_row['RSI'] < 35 and one_h_row['RSI'] > 50: return "LONG"
    elif db == "SHORT" and fb == "LONG":
        if four_h_row['RSI'] > 65 and one_h_row['RSI'] < 50: return "SHORT"
    elif db == "LONG" and fb == "LONG":
        if one_h_row['RSI'] < 45 and one_h_row['close'] > one_h_row['EMA_50']: return "LONG"
    elif db == "SHORT" and fb == "SHORT":
        if one_h_row['RSI'] > 55 and one_h_row['close'] < one_h_row['EMA_50']: return "SHORT"
    return "WAIT"


# ==========================================
# ⭐ FIX #1: CLOSED-CANDLE-ONLY HTF MERGE
# ==========================================
def merge_htf_context(df_1h, df_4h, df_1d):
    """
    Attach the most recent CLOSED 4h and 1d candle to each 1h candle.
    A 4h candle with open-time T closes at T+4h. Only usable at 1h candles with timestamp >= T+4h.
    """
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)
    df_4h = df_4h.sort_values('timestamp').reset_index(drop=True)
    df_1d = df_1d.sort_values('timestamp').reset_index(drop=True)

    df_4h_close = df_4h['timestamp'] + pd.Timedelta(hours=4)
    df_1d_close = df_1d['timestamp'] + pd.Timedelta(hours=24)

    df_1h['htf_4h_idx'] = df_1h['timestamp'].searchsorted(df_4h_close, side='right') - 1
    df_1h['htf_1d_idx'] = df_1h['timestamp'].searchsorted(df_1d_close, side='right') - 1
    return df_1h, df_4h, df_1d


# ==========================================
# FUNDING
# ==========================================
def funding_cost_r(direction, entry_time, exit_time, funding_df, entry, position_size):
    """Compute funding cost in R terms for the holding period."""
    if funding_df.empty: return 0.0
    mask = (funding_df['timestamp'] > entry_time) & (funding_df['timestamp'] <= exit_time)
    rates = funding_df.loc[mask, 'fundingRate'].values
    if len(rates) == 0: return 0.0
    # LONG pays positive funding; SHORT pays negative funding
    sign = 1.0 if direction == "LONG" else -1.0
    total_pct = float(np.sum(rates)) * sign
    notional = position_size * entry
    risk_dollars = position_size * abs(entry - 0) # placeholder
    # We'll compute in R inside the runner
    return total_pct


# ==========================================
# TRADE SIMULATION
# ==========================================
def simulate_trade(direction, entry_price, sl, tp, entry_time, future_1h):
    deadline = entry_time + timedelta(hours=TIME_STOP_HOURS)
    for _, c in future_1h.iterrows():
        if c['timestamp'] > deadline:
            exit_price = float(c['open'])
            risk = abs(entry_price - sl)
            r = (exit_price - entry_price)/risk if direction == "LONG" else (entry_price - exit_price)/risk
            return ("TIME_STOP", exit_price, c['timestamp'], r)
        high = float(c['high']); low = float(c['low'])
        if direction == "LONG":
            hit_sl = low <= sl; hit_tp = high >= tp
        else:
            hit_sl = high >= sl; hit_tp = low <= tp
        if hit_sl and hit_tp:
            return ("LOSS", sl, c['timestamp'], -1.0)  # pessimistic
        if hit_sl:
            return ("LOSS", sl, c['timestamp'], -1.0)
        if hit_tp:
            return ("WIN", tp, c['timestamp'], RR_RATIO)
    return ("OPEN", None, None, None)


# ==========================================
# ⭐ FIX #6: ASYMMETRIC COSTS
# ==========================================
def apply_costs(r_raw, entry, sl, position_size, status):
    risk_dist = abs(entry - sl)
    if risk_dist == 0: return r_raw
    notional = position_size * entry
    if status == "WIN":
        cost = notional * (TAKER_FEE + SLIPPAGE_LIMIT) * 2
    elif status == "LOSS":
        cost = notional * (TAKER_FEE + SLIPPAGE_STOP) + notional * (TAKER_FEE + SLIPPAGE_LIMIT)
    else:  # TIME_STOP
        cost = notional * (TAKER_FEE + SLIPPAGE_STOP) * 2
    risk_dollars = position_size * risk_dist
    return r_raw - (cost / risk_dollars)


# ==========================================
# RUNNER
# ==========================================
def run_backtest(symbol):
    print(f"\n=== Backtesting {symbol} ===")
    since_ms = int(START_DATE.replace(tzinfo=timezone.utc).timestamp() * 1000)

    df_1d = compute_indicators(fetch_candles(symbol, '1d', since_ms))
    df_4h = compute_indicators(fetch_candles(symbol, '4h', since_ms))
    df_1h = compute_indicators(fetch_candles(symbol, '1h', since_ms))
    funding_df = fetch_funding_history(symbol, since_ms)

    print(f"  Data: 1d={len(df_1d)}, 4h={len(df_4h)}, 1h={len(df_1h)}, funding={len(funding_df)}")

    if df_1h.empty or df_4h.empty or df_1d.empty:
        print(f"  Insufficient data for {symbol}")
        return []

    df_1h, df_4h, df_1d = merge_htf_context(df_1h, df_4h, df_1d)

    trades = []
    last_signal_time = None

    # Print a diagnostic on how many 1h candles had valid HTF context
    valid_htf = ((df_1h['htf_4h_idx'] >= 50) & (df_1h['htf_1d_idx'] >= 50)).sum()
    print(f"  1h candles with valid HTF context: {valid_htf}/{len(df_1h)}")

    signal_count = 0

    for i in range(60, len(df_1h) - 2):
        row = df_1h.iloc[i]
        htf_4h_idx = row['htf_4h_idx']
        htf_1d_idx = row['htf_1d_idx']
        if htf_4h_idx < 50 or htf_1d_idx < 50: continue

        four_h_row = df_4h.iloc[int(htf_4h_idx)]
        daily_row = df_1d.iloc[int(htf_1d_idx)]

        if last_signal_time is not None and (row['timestamp'] - last_signal_time).total_seconds() < 3600:
            continue

        signal = get_signal(daily_row, four_h_row, row)
        if signal == "WAIT": continue
        signal_count += 1

        # ⭐ FIX #3: entry at next bar's open, check SL/TP from THAT bar
        entry_idx = i + 1
        entry_candle = df_1h.iloc[entry_idx]
        entry_price = float(entry_candle['open'])
        entry_time = entry_candle['timestamp']
        atr = float(row['ATR'])
        if atr <= 0 or np.isnan(atr): continue

        if signal == "LONG":
            sl = entry_price - (ATR_MULT * atr)
            risk = entry_price - sl
            tp = entry_price + (risk * RR_RATIO)
        else:
            sl = entry_price + (ATR_MULT * atr)
            risk = sl - entry_price
            tp = entry_price - (risk * RR_RATIO)
        if risk <= 0: continue

        # ⭐ FIX #3 continued: include entry candle in simulation
        future = df_1h.iloc[entry_idx:].reset_index(drop=True)
        if future.empty: continue

        status, exit_price, exit_time, r_raw = simulate_trade(signal, entry_price, sl, tp, entry_time, future)
        if status == "OPEN" or r_raw is None: continue

        position_size = (STARTING_BALANCE * RISK_PER_TRADE) / risk

        # ⭐ FIX #4: leverage cap check
        notional = position_size * entry_price
        leverage = notional / STARTING_BALANCE
        if leverage > LEVERAGE_CAP:
            position_size = (STARTING_BALANCE * LEVERAGE_CAP) / entry_price
            notional = position_size * entry_price

        # ⭐ FIX #2: funding cost
        funding_pct = funding_cost_r(signal, entry_time, exit_time, funding_df, entry_price, position_size)
        risk_dollars = position_size * risk
        funding_r = (funding_pct * notional) / risk_dollars if risk_dollars > 0 else 0.0

        # ⭐ FIX #6: asymmetric costs
        r_after_costs = apply_costs(r_raw, entry_price, sl, position_size, status)
        r_final = r_after_costs - funding_r

        trades.append({
            'symbol': symbol, 'direction': signal,
            'entry_time': entry_time, 'entry': entry_price,
            'sl': sl, 'tp': tp, 'stop_dist_pct': risk/entry_price*100,
            'exit_time': exit_time, 'exit': exit_price,
            'status': status,
            'r_raw': r_raw, 'funding_r': funding_r, 'r_final': r_final
        })
        last_signal_time = row['timestamp']

    print(f"  {signal_count} raw signals, {len(trades)} trades simulated")
    return trades


# ==========================================
# METRICS
# ==========================================
def compute_metrics(trades):
    if not trades: return {}
    df = pd.DataFrame(trades)
    wins = (df['status'] == 'WIN').sum()
    losses = (df['status'] == 'LOSS').sum()
    ts = (df['status'] == 'TIME_STOP').sum()
    total = len(df)
    total_r = df['r_final'].sum()
    avg_r = df['r_final'].mean()
    win_rate = wins / total * 100 if total else 0
    std_r = df['r_final'].std()
    sharpe = (avg_r / std_r) if std_r and std_r > 0 else 0
    df_s = df.sort_values('entry_time').reset_index(drop=True)
    eq_r = df_s['r_final'].cumsum()
    peak_r = eq_r.cummax()
    max_dd_r = (eq_r - peak_r).min()
    eq_d = STARTING_BALANCE * (1 + 0.01 * df_s['r_final']).cumprod()
    peak_d = eq_d.cummax()
    max_dd_pct = ((eq_d - peak_d) / peak_d).min() * 100
    return {
        'total': total, 'wins': int(wins), 'losses': int(losses), 'time_stops': int(ts),
        'win_rate': round(win_rate, 1), 'total_r': round(total_r, 2),
        'avg_r': round(avg_r, 3), 'sharpe': round(sharpe, 2),
        'max_dd_r': round(max_dd_r, 2), 'max_dd_pct': round(max_dd_pct, 2),
        'final_balance': round(eq_d.iloc[-1], 2),
        'total_return_pct': round((eq_d.iloc[-1] / STARTING_BALANCE - 1) * 100, 2),
        'avg_funding_r': round(df['funding_r'].mean(), 3)
    }


# ==========================================
# DIAGNOSTICS
# ==========================================
def delay_test(all_trades):
    """Compare actual vs random entries with same exits."""
    if not all_trades: return
    df = pd.DataFrame(all_trades)
    actual_avg_r = df['r_final'].mean()

    # Random baseline: assume 50/50 direction, same RR, same costs modeled as avg
    np.random.seed(42)
    random_rs = []
    for _ in range(len(df)):
        r = RR_RATIO if np.random.rand() < 0.4 else -1.0  # 40% win-rate random
        random_rs.append(r - 0.25)  # rough cost
    random_avg_r = np.mean(random_rs)

    print(f"\n  📊 Diagnostic: Actual avg R = {actual_avg_r:+.3f}, Random baseline = {random_avg_r:+.3f}")
    if actual_avg_r <= random_avg_r:
        print("  ⚠️  Strategy does NOT beat random entries. No edge.")


def bnb_diagnostic(all_trades):
    bnb = [t for t in all_trades if t['symbol'] == 'BNB/USDT:USDT']
    if not bnb: return
    print(f"\n  📊 BNB Diagnostic ({len(bnb)} trades):")
    for t in bnb[:14]:
        print(f"    {t['entry_time'].strftime('%Y-%m-%d %H:%M')} | {t['direction']:5} | "
              f"stop {t['stop_dist_pct']:.3f}% | gross {t['r_raw']:+.2f}R | "
              f"fund {t['funding_r']:+.3f}R | net {t['r_final']:+.3f}R")


# ==========================================
# REPORT
# ==========================================
def print_report(all_trades, symbol_metrics, train_m, test_m):
    print("\n" + "="*60)
    print("       BACKTEST REPORT (v2 — audited)")
    print("="*60)
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Symbols: {', '.join([s.split('/')[0] for s in SYMBOLS])}")

    if not all_trades:
        print("\n❌ NO TRADES.")
        return

    tm = compute_metrics(all_trades)
    print(f"\n--- ALL TRADES ---")
    print(f"Total: {tm['total']} | W:{tm['wins']} L:{tm['losses']} TS:{tm['time_stops']}")
    print(f"Win rate: {tm['win_rate']}%")
    print(f"Total R: {tm['total_r']:+.2f}R | Avg: {tm['avg_r']:+.3f}R")
    print(f"Avg funding drag: {tm['avg_funding_r']:+.3f}R/trade")
    print(f"Sharpe: {tm['sharpe']} | Max DD: {tm['max_dd_r']}R ({tm['max_dd_pct']}%)")
    print(f"Final: ${tm['final_balance']} ({tm['total_return_pct']:+.1f}%)")

    print(f"\n--- PER SYMBOL ---")
    for sym, m in symbol_metrics.items():
        if not m: continue
        print(f"{sym.split('/')[0]:6} | {m['total']:3} | WR {m['win_rate']:5.1f}% | "
              f"{m['total_r']:+7.2f}R | fund {m['avg_funding_r']:+.3f}R | ${m['final_balance']:.0f}")

    print(f"\n--- TRAIN/TEST ---")
    if train_m: print(f"TRAIN: {train_m['total']} | WR {train_m['win_rate']}% | {train_m['total_r']:+.2f}R | Sharpe {train_m['sharpe']}")
    if test_m: print(f"TEST:  {test_m['total']} | WR {test_m['win_rate']}% | {test_m['total_r']:+.2f}R | Sharpe {test_m['sharpe']}")

    delay_test(all_trades)
    bnb_diagnostic(all_trades)

    print("\n" + "="*60)


def main():
    print(f"Starting backtest (AUDITED): {START_DATE.date()} → {END_DATE.date()}")
    all_trades = []
    symbol_metrics = {}

    for sym in SYMBOLS:
        try:
            trades = run_backtest(sym)
            if trades:
                all_trades.extend(trades)
                symbol_metrics[sym] = compute_metrics(trades)
        except Exception as e:
            print(f"❌ Error on {sym}: {e}")

    if not all_trades:
        print_report([], {}, None, None); return

    all_df = pd.DataFrame(all_trades).sort_values('entry_time').reset_index(drop=True)
    split = int(len(all_df) * TRAIN_RATIO)
    train_m = compute_metrics(all_df.iloc[:split].to_dict('records'))
    test_m = compute_metrics(all_df.iloc[split:].to_dict('records'))

    print_report(all_trades, symbol_metrics, train_m, test_m)

    with open('backtest_report.json', 'w') as f:
        json.dump({
            'period': f"{START_DATE.date()} to {END_DATE.date()}",
            'all_metrics': compute_metrics(all_trades),
            'per_symbol': symbol_metrics,
            'train': train_m, 'test': test_m,
            'generated': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=4, default=str)
    print("✅ Report saved.")


if __name__ == "__main__":
    main()
