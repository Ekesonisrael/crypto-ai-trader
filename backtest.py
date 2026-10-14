import ccxt
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timezone, timedelta

# ==========================================
# CONFIG
# ==========================================
SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT']

# Historical window (2 years)
END_DATE = datetime.now(timezone.utc).replace(tzinfo=None)
START_DATE = END_DATE - timedelta(days=730)

# Costs
TAKER_FEE = 0.0005
SLIPPAGE = 0.0005

# Strategy params (must match live bot)
ATR_MULT = 1.5
RR_RATIO = 2.0
ADX_MIN = 20
RISK_PER_TRADE = 0.01  # 1% of balance
STARTING_BALANCE = 10000.0
TIME_STOP_HOURS = 168  # 7 days

# Train/test split (70/30)
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
            print(f"    Error: {e}. Retrying in 3s...")
            import time; time.sleep(3)
            continue
        if not batch:
            break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= exchange.milliseconds() - 60000:
            break
        if len(batch) < limit_per_call:
            break
        cursor = last_ts + 1

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    # Drop currently-forming candle
    now_ms = exchange.milliseconds()
    df = df[df['timestamp'].astype('int64') // 10**6 < now_ms - 60000]
    return df.drop_duplicates(subset='timestamp').reset_index(drop=True)


# ==========================================
# INDICATOR CALCULATION (same as live bot)
# ==========================================
def compute_indicators(df):
    df = df.copy()
    # RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # EMAs
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    # ATR 14
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    # ADX 14
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
# SIGNAL LOGIC (must match live)
# ==========================================
def get_signal(daily_row, four_h_row, one_h_row):
    daily_bias = "LONG" if daily_row['EMA_20'] > daily_row['EMA_50'] else "SHORT"
    four_h_bias = "LONG" if four_h_row['EMA_20'] > four_h_row['EMA_50'] else "SHORT"
    adx = one_h_row['ADX']

    if pd.isna(adx) or adx < ADX_MIN:
        return "WAIT"

    if daily_bias == "LONG" and four_h_bias == "SHORT":
        if four_h_row['RSI'] < 35 and one_h_row['RSI'] > 50:
            return "LONG"
    elif daily_bias == "SHORT" and four_h_bias == "LONG":
        if four_h_row['RSI'] > 65 and one_h_row['RSI'] < 50:
            return "SHORT"
    elif daily_bias == "LONG" and four_h_bias == "LONG":
        if one_h_row['RSI'] < 45 and one_h_row['close'] > one_h_row['EMA_50']:
            return "LONG"
    elif daily_bias == "SHORT" and four_h_bias == "SHORT":
        if one_h_row['RSI'] > 55 and one_h_row['close'] < one_h_row['EMA_50']:
            return "SHORT"
    return "WAIT"


def merge_htf_context(df_1h, df_4h, df_1d):
    """For each 1h candle, attach the most recent CLOSED 4h and 1d candle data."""
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)
    df_4h = df_4h.sort_values('timestamp').reset_index(drop=True)
    df_1d = df_1d.sort_values('timestamp').reset_index(drop=True)

    df_1h['htf_4h_idx'] = df_1h['timestamp'].searchsorted(df_4h['timestamp'], side='right') - 1
    df_1h['htf_1d_idx'] = df_1h['timestamp'].searchsorted(df_1d['timestamp'], side='right') - 1
    return df_1h, df_4h, df_1d


# ==========================================
# TRADE SIMULATION
# ==========================================
def simulate_trade(direction, entry_price, sl, tp, entry_time, future_1h):
    """Walk forward. Returns (status, exit_price, exit_time, r_multiple_raw)."""
    deadline = entry_time + timedelta(hours=TIME_STOP_HOURS)
    for _, c in future_1h.iterrows():
        if c['timestamp'] > deadline:
            exit_price = float(c['open'])
            risk = abs(entry_price - sl)
            r = (exit_price - entry_price)/risk if direction == "LONG" else (entry_price - exit_price)/risk
            return ("TIME_STOP", exit_price, c['timestamp'], r)

        high = float(c['high']); low = float(c['low'])
        if direction == "LONG":
            hit_sl = low <= sl
            hit_tp = high >= tp
        else:
            hit_sl = high >= sl
            hit_tp = low <= tp

        # Pessimistic: if both, count as LOSS
        if hit_sl and hit_tp:
            return ("LOSS", sl, c['timestamp'], -1.0)
        if hit_sl:
            return ("LOSS", sl, c['timestamp'], -1.0)
        if hit_tp:
            return ("WIN", tp, c['timestamp'], RR_RATIO)

    return ("OPEN", None, None, None)


def apply_costs(r_mult, entry, sl, position_size):
    risk_dist = abs(entry - sl)
    if risk_dist == 0: return r_mult
    notional = position_size * entry
    cost = notional * (TAKER_FEE + SLIPPAGE) * 2
    risk_dollars = position_size * risk_dist
    return r_mult - (cost / risk_dollars)


# ==========================================
# BACKTEST RUNNER
# ==========================================
def run_backtest(symbol):
    print(f"\n=== Backtesting {symbol} ===")
    since_ms = int(START_DATE.replace(tzinfo=timezone.utc).timestamp() * 1000)

    df_1d = compute_indicators(fetch_candles(symbol, '1d', since_ms))
    df_4h = compute_indicators(fetch_candles(symbol, '4h', since_ms))
    df_1h = compute_indicators(fetch_candles(symbol, '1h', since_ms))

    if df_1h.empty or df_4h.empty or df_1d.empty:
        print(f"  Insufficient data for {symbol}")
        return []

    df_1h, df_4h, df_1d = merge_htf_context(df_1h, df_4h, df_1d)

    trades = []
    last_signal_time = None

    for i in range(60, len(df_1h) - 2):
        row = df_1h.iloc[i]

        htf_4h_idx = row['htf_4h_idx']
        htf_1d_idx = row['htf_1d_idx']
        if htf_4h_idx < 50 or htf_1d_idx < 50:
            continue

        four_h_row = df_4h.iloc[int(htf_4h_idx)]
        daily_row = df_1d.iloc[int(htf_1d_idx)]

        # Cooldown: 1 hour minimum between trades on same symbol
        if last_signal_time is not None and (row['timestamp'] - last_signal_time).total_seconds() < 3600:
            continue

        signal = get_signal(daily_row, four_h_row, row)
        if signal == "WAIT":
            continue

        # Signal fires at close of candle i. Enter at open of candle i+1.
        entry_candle = df_1h.iloc[i + 1]
        entry_price = float(entry_candle['open'])
        entry_time = entry_candle['timestamp']
        atr = float(row['ATR'])
        if atr <= 0 or np.isnan(atr):
            continue

        if signal == "LONG":
            sl = entry_price - (ATR_MULT * atr)
            risk = entry_price - sl
            tp = entry_price + (risk * RR_RATIO)
        else:
            sl = entry_price + (ATR_MULT * atr)
            risk = sl - entry_price
            tp = entry_price - (risk * RR_RATIO)

        if risk <= 0:
            continue

        # Simulate from candle i+2 onward (i+1 is entry candle)
        future = df_1h.iloc[i + 2:].reset_index(drop=True)
        if future.empty:
            continue

        status, exit_price, exit_time, r_raw = simulate_trade(signal, entry_price, sl, tp, entry_time, future)

        if status == "OPEN" or r_raw is None:
            continue

        position_size = (STARTING_BALANCE * RISK_PER_TRADE) / risk
        r_final = apply_costs(r_raw, entry_price, sl, position_size)

        trades.append({
            'symbol': symbol,
            'direction': signal,
            'entry_time': entry_time,
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'exit_time': exit_time,
            'exit': exit_price,
            'status': status,
            'r_raw': r_raw,
            'r_final': r_final
        })

        last_signal_time = row['timestamp']

    print(f"  {len(trades)} trades simulated.")
    return trades


# ==========================================
# METRICS
# ==========================================
def compute_metrics(trades):
    if not trades:
        return {}
    df = pd.DataFrame(trades)

    wins = (df['status'] == 'WIN').sum()
    losses = (df['status'] == 'LOSS').sum()
    time_stops = (df['status'] == 'TIME_STOP').sum()
    total = len(df)

    total_r = df['r_final'].sum()
    avg_r = df['r_final'].mean()

    win_rate = wins / total * 100 if total > 0 else 0
    # Expectancy per trade
    expectancy = avg_r

    # Sharpe-ish (mean R / std R)
    std_r = df['r_final'].std()
    sharpe = (avg_r / std_r) if std_r and std_r > 0 else 0

    # Max drawdown on equity curve (in R terms)
    df_sorted = df.sort_values('entry_time').reset_index(drop=True)
    equity = df_sorted['r_final'].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min()

    # Equity in $ terms (approximate using fixed 1% risk)
    equity_dollars = STARTING_BALANCE * (1 + 0.01 * df_sorted['r_final']).cumprod()
    peak_d = equity_dollars.cummax()
    dd_d = (equity_dollars - peak_d) / peak_d
    max_dd_pct = dd_d.min() * 100

    final_balance = equity_dollars.iloc[-1]
    total_return_pct = (final_balance / STARTING_BALANCE - 1) * 100

    return {
        'total': total,
        'wins': int(wins),
        'losses': int(losses),
        'time_stops': int(time_stops),
        'win_rate': round(win_rate, 1),
        'total_r': round(total_r, 2),
        'avg_r': round(avg_r, 3),
        'expectancy': round(expectancy, 3),
        'sharpe': round(sharpe, 2),
        'max_dd_r': round(max_dd, 2),
        'max_dd_pct': round(max_dd_pct, 2),
        'final_balance': round(final_balance, 2),
        'total_return_pct': round(total_return_pct, 2)
    }


def print_report(all_trades, symbol_metrics, train_metrics, test_metrics, buy_and_hold):
    print("\n" + "=" * 60)
    print("       BACKTEST REPORT")
    print("=" * 60)
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Symbols: {', '.join([s.split('/')[0] for s in SYMBOLS])}")
    print(f"Risk/trade: {RISK_PER_TRADE*100}%  |  R:R 1:{RR_RATIO}  |  ADX min: {ADX_MIN}")

    if not all_trades:
        print("\n❌ NO TRADES TRIGGERED. Strategy is too strict or data range too short.")
        return

    total_m = compute_metrics(all_trades)
    print(f"\n--- ALL TRADES ---")
    print(f"Total: {total_m['total']} | W:{total_m['wins']} L:{total_m['losses']} TS:{total_m['time_stops']}")
    print(f"Win rate: {total_m['win_rate']}%")
    print(f"Total R: {total_m['total_r']:+.2f}R | Avg: {total_m['avg_r']:+.3f}R | Expectancy: {total_m['expectancy']:+.3f}R")
    print(f"Sharpe (R): {total_m['sharpe']} | Max DD: {total_m['max_dd_r']}R ({total_m['max_dd_pct']}%)")
    print(f"Final balance: ${total_m['final_balance']} ({total_m['total_return_pct']:+.1f}%)")

    print(f"\n--- PER SYMBOL ---")
    for sym, m in symbol_metrics.items():
        if not m: continue
        print(f"{sym.split('/')[0]:6} | {m['total']:3} trades | WR {m['win_rate']:5.1f}% | {m['total_r']:+7.2f}R | ${m['final_balance']:.0f}")

    print(f"\n--- TRAIN/TEST SPLIT ---")
    if train_metrics:
        print(f"TRAIN: {train_metrics['total']} trades | WR {train_metrics['win_rate']}% | {train_metrics['total_r']:+.2f}R | Sharpe {train_metrics['sharpe']}")
    if test_metrics:
        print(f"TEST:  {test_metrics['total']} trades | WR {test_metrics['win_rate']}% | {test_metrics['total_r']:+.2f}R | Sharpe {test_metrics['sharpe']}")

    print(f"\n--- BASELINE ---")
    print(f"Buy & Hold BTC: {buy_and_hold:+.1f}%")

    # Verdict
    print(f"\n--- VERDICT ---")
    if test_metrics and test_metrics['total'] >= 10:
        if test_metrics['expectancy'] > 0.1 and test_metrics['sharpe'] > 0.3:
            print("✅ PROMISING: Positive expectancy in out-of-sample test.")
        elif test_metrics['expectancy'] > 0:
            print("⚠️  MARGINAL: Small positive edge, but not convincing.")
        else:
            print("❌ NEGATIVE EDGE: Do NOT deploy. Strategy needs rework.")
    else:
        print("⚠️  INSUFFICIENT SAMPLE: Need more trades to judge.")
    print("=" * 60)


def main():
    print(f"Starting backtest: {START_DATE.date()} → {END_DATE.date()}")
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
        print_report([], {}, None, None, 0.0)
        return

    all_df = pd.DataFrame(all_trades).sort_values('entry_time').reset_index(drop=True)

    # Train/test split
    split_idx = int(len(all_df) * TRAIN_RATIO)
    train_trades = all_df.iloc[:split_idx].to_dict('records')
    test_trades = all_df.iloc[split_idx:].to_dict('records')

    train_metrics = compute_metrics(train_trades) if train_trades else None
    test_metrics = compute_metrics(test_trades) if test_trades else None

    # Buy & hold BTC baseline
    btc_1h = compute_indicators(fetch_candles('BTC/USDT:USDT', '1h', int(START_DATE.replace(tzinfo=timezone.utc).timestamp() * 1000)))
    bh = 0.0
    if not btc_1h.empty:
        bh = (btc_1h['close'].iloc[-1] / btc_1h['close'].iloc[0] - 1) * 100

    print_report(all_trades, symbol_metrics, train_metrics, test_metrics, bh)

    # Save report
    report = {
        'period': f"{START_DATE.date()} to {END_DATE.date()}",
        'all_metrics': compute_metrics(all_trades),
        'per_symbol': symbol_metrics,
        'train': train_metrics,
        'test': test_metrics,
        'buy_hold_btc_pct': round(bh, 2),
        'generated': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    with open('backtest_report.json', 'w') as f:
        json.dump(report, f, indent=4, default=str)
    print("\n✅ Report saved to backtest_report.json")


if __name__ == "__main__":
    main()
