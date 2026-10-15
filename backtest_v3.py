#!/usr/bin/env python3
"""
backtest.py  (v3 - updated with Binance Vision 4-year data pipeline)

Multi-timeframe (1d / 4h / 1h) swing strategy backtester for OKX & Binance USDT perpetuals.

USAGE
    python backtest.py --selftest        # offline logic tests (run this FIRST, no network needed)
    python backtest.py                   # full 4-year run on Binance Vision / OKX data
    python backtest.py --days 365 --symbols BTC ETH --no-cache
    python backtest.py --max-cost-r 0.25 # only take trades whose round-trip cost <= 0.25R

DEPENDENCIES: numpy, pandas, requests   (ccxt is no longer needed)
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT']
BACKTEST_DAYS = 1460

# --- Strategy parameters ---
ATR_MULT = 1.5
RR_RATIO = 2.0
ADX_MIN = 20
RSI_4H_OVERSOLD, RSI_4H_OVERBOUGHT = 35, 65
RSI_1H_PULLBACK_LONG, RSI_1H_PULLBACK_SHORT = 45, 55
TIME_STOP_HOURS = 168
MIN_STOP_PCT = 1.5    # reject signals where the ATR stop is tighter than this % of entry

# --- Indicators ---
USE_WILDER = True      # True = standard Wilder smoothing (TradingView/TA-Lib style).
MIN_BARS = 150         # warm-up bars required per timeframe before signals are trusted

# --- Sizing ---
STARTING_BALANCE = 10_000.0
RISK_PER_TRADE = 0.01      # fixed-dollar risk = 1% of STARTING balance (non-compounding)
LEVERAGE_CAP = 3.0
MAX_CONCURRENT = None      # e.g. 3 to cap simultaneous positions across symbols (None = no cap)
MAX_COST_R = None          # skip signals whose estimated round-trip cost exceeds this many R

# --- Costs (fractions of notional) ---
TAKER_FEE = 0.0005         # OKX/Binance perp taker, regular tier
TP_FEE = 0.0005            # conservative: treat TP limit fills as taker (maker is lower)
SLIPPAGE_ENTRY = 0.0003
SLIPPAGE_TP = 0.0003
SLIPPAGE_STOP = 0.0015     # market-like stop fills

# --- Funding ---
FALLBACK_FUNDING_8H = 0.0001   # 0.01% per 8h baseline, used only where no data exists

# --- Evaluation ---
TRAIN_RATIO = 0.70             # time-based split of the test window
MC_ITERATIONS = 5000
BOOTSTRAP_ITERATIONS = 5000
RNG_SEED = 42

# --- Pass/fail gates ---
GATES = {
    'min_oos_trades': 100,        # fewer than this => INCONCLUSIVE, not pass/fail
    'min_oos_mean_r': 0.10,       # out-of-sample mean net R per trade
    'ci_lower_gt_zero': True,     # bootstrap 95% CI lower bound of OOS mean R must be > 0
    'baseline_p_max': 0.05,       # must beat random-entry baseline (one-sided p)
    'stress_cost_mult': 1.5,      # OOS mean R must stay > 0 at this multiple of trading costs
    'max_dd_pct_of_start': 25.0,  # full-sample realized max drawdown, % of starting balance
}

# --- Data & Endpoints ---
CACHE_DIR = '.bt_cache'
CACHE_MAX_AGE_HOURS = 12
REQUEST_SLEEP = 0.12           # OKX history-candles limit is 20 req / 2s per IP
OKX_BASE = 'https://www.okx.com'
BINANCE_VISION_BASE = 'https://data.binance.vision'
HOUR = 3_600_000
DAY = 86_400_000
ALLOW_SHORT_HISTORY = False

# OKX bar codes
TF = {'1h': ('1H', HOUR), '4h': ('4H', 4 * HOUR), '1d': ('1Dutc', DAY)}

# status codes
S_NONE, S_WIN, S_LOSS, S_TIME, S_OPEN = 0, 1, 2, 3, 4
S_NAME = {S_WIN: 'WIN', S_LOSS: 'LOSS', S_TIME: 'TIME_STOP'}


class DataError(Exception):
    pass


class FatalAPIError(Exception):
    pass


# =============================================================================
# HTTP + DATA FETCHING
# =============================================================================
def _http_get_json(url, params, timeout=30):
    """Single place that touches the network (monkeypatched in --selftest)."""
    import requests
    r = requests.get(url, params=params, timeout=timeout,
                     headers={'User-Agent': 'backtest-v3'})
    r.raise_for_status()
    return r.json()


def okx_get(path, params, retries=8):
    last = None
    for k in range(retries):
        try:
            j = _http_get_json(OKX_BASE + path, params)
        except Exception as e:
            last = e
            time.sleep(min(30, 0.5 * 2 ** k))
            continue
        code = str(j.get('code'))
        if code == '0':
            time.sleep(REQUEST_SLEEP)
            return j.get('data', [])
        if code in ('50011', '50013'):
            last = RuntimeError(f"OKX {code}: {j.get('msg')}")
            time.sleep(min(30, 1.0 * 2 ** k))
            continue
        raise FatalAPIError(f"OKX {code}: {j.get('msg')}  ({path} {params})")
    raise DataError(f"OKX request failed after {retries} tries ({path} {params}): {last}")


def inst_id_of(symbol):
    """'BTC/USDT:USDT' -> 'BTC-USDT-SWAP'"""
    pair = symbol.split(':')[0]
    base, quote = pair.split('/')
    return f"{base}-{quote}-SWAP"


def fetch_candles(inst_id, tf, start_ms, end_ms, use_cache=True):
    """Full-history candle download using OKX /market/history-candles."""
    bar, step = TF[tf]
    cache_path = os.path.join(CACHE_DIR, f"{inst_id}_{bar}.csv")
    if use_cache and os.path.exists(cache_path):
        age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_h <= CACHE_MAX_AGE_HOURS:
            c = pd.read_csv(cache_path)
            if len(c) and int(c['ts'].iloc[0]) <= start_ms + step:
                return _trim_closed(c, start_ms, end_ms, step)

    rows = {}
    after = None
    while True:
        params = {'instId': inst_id, 'bar': bar, 'limit': '100'}
        if after is not None:
            params['after'] = str(after)
        data = okx_get('/api/v5/market/history-candles', params)
        if not data:
            break
        for r in data:
            if len(r) >= 9 and str(r[8]) != '1':
                continue
            ts = int(r[0])
            rows[ts] = (ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
        batch_oldest = min(int(r[0]) for r in data)
        if after is not None and batch_oldest >= after:
            break
        after = batch_oldest
        if batch_oldest <= start_ms:
            break
    if not rows:
        return pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df = pd.DataFrame(sorted(rows.values()),
                      columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return _trim_closed(df, start_ms, end_ms, step)


def fetch_candles_binance_vision(symbol, tf, start_ms, end_ms):
    """Download monthly CSV dumps from data.binance.vision (public, no auth)."""
    import io, zipfile, requests as _req
    base = symbol.split('/')[0] + 'USDT'
    rows = []
    start_dt = datetime.fromtimestamp(start_ms / 1000, timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, timezone.utc)
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        url = (f"{BINANCE_VISION_BASE}/data/futures/um/monthly/klines/"
               f"{base}/{tf}/{base}-{tf}-{y}-{m:02d}.zip")
        try:
            r = _req.get(url, timeout=90)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
                try:
                    int(df.iloc[0, 0])
                except (ValueError, TypeError):
                    df = df.iloc[1:].reset_index(drop=True)
                df = df.iloc[:, :6]
                df.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
                for c in ['ts', 'open', 'high', 'low', 'close', 'volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.dropna(subset=['ts'])
                df['ts'] = df['ts'].astype('int64')
                rows.append(df)
                time.sleep(0.05)
        except Exception as e:
            print(f"    Binance Vision {base} {tf} {y}-{m:02d}: {e}")
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
    if not rows:
        return pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    out = pd.concat(rows).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    return _trim_closed(out, start_ms, end_ms, TF[tf][1])


def fetch_funding_binance_vision(base, start_ms, end_ms):
    """Download monthly funding-rate dumps from data.binance.vision."""
    import io, zipfile, requests as _req
    rows = {}
    start_dt = datetime.fromtimestamp(start_ms / 1000, timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, timezone.utc)
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        url = (f"{BINANCE_VISION_BASE}/data/futures/um/monthly/fundingRate/"
               f"{base}USDT/{base}USDT-fundingRate-{y}-{m:02d}.zip")
        try:
            r = _req.get(url, timeout=60)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]))
                df.columns = [c.lower().strip() for c in df.columns]
                for _, row in df.iterrows():
                    try:
                        t = int(row['calc_time'])
                        rate = float(row['last_funding_rate'])
                        if start_ms <= t <= end_ms:
                            rows[t] = rate
                    except (ValueError, KeyError, TypeError):
                        continue
                time.sleep(0.05)
        except Exception:
            pass
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
    return sorted(rows.items())


def _trim_closed(df, start_ms, end_ms, step):
    df = df.drop_duplicates('ts').sort_values('ts')
    df = df[(df['ts'] >= start_ms) & (df['ts'] + step <= end_ms)]
    return df.reset_index(drop=True)


def fetch_funding_okx(inst_id, start_ms):
    """OKX funding-rate history."""
    rows = {}
    after = None
    while True:
        params = {'instId': inst_id, 'limit': '100'}
        if after is not None:
            params['after'] = str(after)
        data = okx_get('/api/v5/public/funding-rate-history', params)
        if not data:
            break
        for r in data:
            t = int(r['fundingTime'])
            rate = r.get('realizedRate')
            if rate in (None, ''):
                rate = r.get('fundingRate')
            if rate in (None, ''):
                continue
            rows[t] = float(rate)
        oldest = min(int(r['fundingTime']) for r in data)
        if after is not None and oldest >= after:
            break
        after = oldest
        if oldest <= start_ms:
            break
    return sorted(rows.items())


def fill_funding_gaps(real, start_ms, end_ms, fallback_rate):
    """Insert assumed 8h funding points wherever the real series has no coverage."""
    out = list(real)
    step = 8 * HOUR
    times = [t for t, _, _ in real]
    if not times:
        t = start_ms - (start_ms % step) + step
        while t <= end_ms:
            out.append((t, fallback_rate, 'assumed'))
            t += step
    else:
        t = times[0] - step
        while t >= start_ms:
            out.append((t, fallback_rate, 'assumed'))
            t -= step
        for a, b in zip(times, times[1:]):
            if b - a > 9 * HOUR:
                t = a + step
                while t < b - HOUR:
                    out.append((t, fallback_rate, 'assumed'))
                    t += step
        t = times[-1] + step
        while t <= end_ms:
            out.append((t, fallback_rate, 'assumed'))
            t += step
    out.sort(key=lambda x: x[0])
    ts = np.array([o[0] for o in out], dtype=np.int64)
    rate = np.array([o[1] for o in out], dtype=float)
    src = [o[2] for o in out]
    return ts, rate, src


def audit_candles(df, tf, label, start_ms, end_ms):
    step = TF[tf][1]
    if df.empty:
        raise DataError(f"{label} {tf}: NO DATA returned")
    first, last = int(df['ts'].iloc[0]), int(df['ts'].iloc[-1])
    expected = (last - first) // step + 1
    missing = expected - len(df)
    span_days = (last - first) / DAY
    f = datetime.fromtimestamp(first / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M')
    l = datetime.fromtimestamp(last / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M')
    flag = '' if missing == 0 else f'  <-- {missing} missing bars ({missing / expected * 100:.2f}%)'
    print(f"    {label:5} {tf:3}: {len(df):6d} bars | {f} -> {l} UTC | span {span_days:7.1f}d{flag}")
    return missing, expected


# =============================================================================
# INDICATORS
# =============================================================================
def _smooth(s, n):
    if USE_WILDER:
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return s.rolling(n).mean()


def compute_indicators(df, n=14):
    df = df.copy()
    c, h, l = df['close'], df['high'], df['low']
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain, avg_loss = _smooth(gain, n), _smooth(loss, n)
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    df['RSI'] = rsi
    df['EMA_20'] = c.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = c.ewm(span=50, adjust=False).mean()
    prev_c = c.shift()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1, skipna=True)
    atr = _smooth(tr, n)
    df['ATR'] = atr
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pdi = 100 * _smooth(pdm, n) / atr
    mdi = 100 * _smooth(mdm, n) / atr
    denom = (pdi + mdi).replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / denom
    df['ADX'] = _smooth(dx, n)
    return df


# =============================================================================
# FRAME BUILD & SIGNALS
# =============================================================================
def build_frame(d1h, d4h, d1d):
    h = d1h[['ts', 'open', 'high', 'low', 'close', 'RSI', 'EMA_20', 'EMA_50', 'ATR', 'ADX']].copy()
    h['bar_close'] = h['ts'] + HOUR
    h['h_n'] = np.arange(len(h))

    f4 = d4h[['ts', 'RSI', 'EMA_20', 'EMA_50']].copy()
    f4.columns = ['h4_ts', 'h4_RSI', 'h4_EMA20', 'h4_EMA50']
    f4['h4_n'] = np.arange(len(f4))
    f4['h4_avail'] = f4['h4_ts'] + 4 * HOUR

    fd = d1d[['ts', 'EMA_20', 'EMA_50']].copy()
    fd.columns = ['d1_ts', 'd1_EMA20', 'd1_EMA50']
    fd['d1_n'] = np.arange(len(fd))
    fd['d1_avail'] = fd['d1_ts'] + DAY

    m = pd.merge_asof(h.sort_values('bar_close'), f4.sort_values('h4_avail'),
                      left_on='bar_close', right_on='h4_avail', direction='backward')
    m = pd.merge_asof(m, fd.sort_values('d1_avail'),
                      left_on='bar_close', right_on='d1_avail', direction='backward')
    m['ctx_ok'] = ((m['h_n'] >= MIN_BARS) & (m['h4_n'] >= MIN_BARS) & (m['d1_n'] >= MIN_BARS))
    return m.reset_index(drop=True)


def compute_signals(m):
    adx_ok = (m['ADX'] >= ADX_MIN) & m['ctx_ok']
    have = m[['RSI', 'h4_RSI', 'd1_EMA20', 'd1_EMA50', 'h4_EMA20', 'h4_EMA50']].notna().all(axis=1)
    valid = (adx_ok & have).to_numpy()
    d_long = (m['d1_EMA20'] > m['d1_EMA50']).to_numpy()
    f_long = (m['h4_EMA20'] > m['h4_EMA50']).to_numpy()
    r1, r4 = m['RSI'].to_numpy(), m['h4_RSI'].to_numpy()
    close, ema50 = m['close'].to_numpy(), m['EMA_50'].to_numpy()
    with np.errstate(invalid='ignore'):
        long_ = ((d_long & ~f_long & (r4 < RSI_4H_OVERSOLD) & (r1 > 50)) |
                 (d_long & f_long & (r1 < RSI_1H_PULLBACK_LONG) & (close > ema50)))
        short_ = ((~d_long & f_long & (r4 > RSI_4H_OVERBOUGHT) & (r1 < 50)) |
                  (~d_long & ~f_long & (r1 > RSI_1H_PULLBACK_SHORT) & (close < ema50)))
    sig = np.zeros(len(m), dtype=np.int8)
    sig[valid & long_] = 1
    sig[valid & short_] = -1
    return sig


# =============================================================================
# TRADE SIMULATION
# =============================================================================
def _sim_one(side, e, sl, tp, o, h, l, ts, deadline_ms, n):
    for j in range(e, n):
        if ts[j] >= deadline_ms:
            return S_TIME, j, o[j]
        if j > e:
            if side == 1 and o[j] <= sl:
                return S_LOSS, j, o[j]
            if side == -1 and o[j] >= sl:
                return S_LOSS, j, o[j]
        if side == 1:
            hit_sl, hit_tp = l[j] <= sl, h[j] >= tp
        else:
            hit_sl, hit_tp = h[j] >= sl, l[j] <= tp
        if hit_sl:
            return S_LOSS, j, sl
        if hit_tp:
            return S_WIN, j, tp
    return S_OPEN, n - 1, float('nan')


class SymbolData:
    def __init__(self, symbol, frame, fund_ts, fund_rate, test_start_ms):
        self.symbol = symbol
        self.m = frame
        self.n = len(frame)
        self.ts = frame['ts'].to_numpy(dtype=np.int64)
        self.o = frame['open'].to_numpy(dtype=float)
        self.h = frame['high'].to_numpy(dtype=float)
        self.l = frame['low'].to_numpy(dtype=float)
        self.atr = frame['ATR'].to_numpy(dtype=float)
        self.adx = frame['ADX'].to_numpy(dtype=float)
        self.ctx_ok = frame['ctx_ok'].to_numpy(dtype=bool)
        self.sig = compute_signals(frame)
        self.test_start_idx = int(np.searchsorted(self.ts, test_start_ms))
        self.fund_ts = fund_ts
        self.fund_cum = np.concatenate([[0.0], np.cumsum(fund_rate)])
        self.out = {1: self._outcomes(1), -1: self._outcomes(-1)}

    def _outcomes(self, side):
        n = self.n
        status = np.zeros(n, dtype=np.int8)
        exit_idx = np.full(n, -1, dtype=np.int64)
        entry_px = np.full(n, np.nan)
        exit_px = np.full(n, np.nan)
        dist = np.full(n, np.nan)
        idxs = np.arange(n)
        ok = (idxs < n - 1) & (idxs >= self.test_start_idx) & self.ctx_ok
        ok &= np.isfinite(self.atr) & (self.atr > 0)
        deadline_add = TIME_STOP_HOURS * HOUR
        o, h, l, ts = self.o, self.h, self.l, self.ts
        for i in np.flatnonzero(ok):
            e = i + 1
            ep = o[e]
            dd = ATR_MULT * self.atr[i]
            if side == 1:
                sl, tp = ep - dd, ep + RR_RATIO * dd
            else:
                sl, tp = ep + dd, ep - RR_RATIO * dd
            if dd <= 0 or sl <= 0 or tp <= 0:
                continue
            st, xj, xp = _sim_one(side, e, sl, tp, o, h, l, ts, ts[e] + deadline_add, n)
            status[i], exit_idx[i], entry_px[i], exit_px[i], dist[i] = st, xj, ep, xp, dd

        done = np.isin(status, (S_WIN, S_LOSS, S_TIME))
        r_raw = np.full(n, np.nan)
        cost_r = np.full(n, np.nan)
        fund_r = np.full(n, np.nan)
        d = done
        r_raw[d] = side * (exit_px[d] - entry_px[d]) / dist[d]
        c_in = TAKER_FEE + SLIPPAGE_ENTRY
        c_out = np.where(status == S_WIN, TP_FEE + SLIPPAGE_TP, TAKER_FEE + SLIPPAGE_STOP)
        cost_r[d] = (c_in * entry_px[d] + c_out[d] * exit_px[d]) / dist[d]
        e_ts = ts[np.minimum(idxs + 1, n - 1)]
        x_ts = ts[np.maximum(exit_idx, 0)]
        k_in = np.searchsorted(self.fund_ts, e_ts, side='right')
        k_out = np.searchsorted(self.fund_ts, x_ts, side='right')
        rate_sum = self.fund_cum[k_out] - self.fund_cum[k_in]
        fund_r[d] = side * rate_sum[d] * entry_px[d] / dist[d]
        d_frac = dist / entry_px
        risk_usd = np.minimum(STARTING_BALANCE * RISK_PER_TRADE,
                              LEVERAGE_CAP * STARTING_BALANCE * d_frac)
        return {'status': status, 'exit_idx': exit_idx, 'entry_px': entry_px, 'exit_px': exit_px,
                'dist': dist, 'r_raw': r_raw, 'cost_r': cost_r, 'fund_r': fund_r,
                'risk_usd': risk_usd, 'd_frac': d_frac}

    def est_cost_r(self, i):
        e = i + 1
        dd = ATR_MULT * self.atr[i]
        ep = self.o[e]
        return ((TAKER_FEE + SLIPPAGE_ENTRY) * ep + (TAKER_FEE + SLIPPAGE_STOP) * ep) / dd


def select_trades(sd, max_cost_r=None):
    rows = []
    next_free = 0
    skipped_cost = skipped_open = 0
    for i in np.flatnonzero(sd.sig != 0):
        if i < next_free or i < sd.test_start_idx or i >= sd.n - 1:
            continue
        side = int(sd.sig[i])
        out = sd.out[side]
        st = int(out['status'][i])
        if st == S_NONE:
            continue
        if st == S_OPEN:
            skipped_open += 1
            continue
        if out['d_frac'][i] * 100 < MIN_STOP_PCT:
            skipped_cost += 1
            continue
        if max_cost_r is not None and sd.est_cost_r(i) > max_cost_r:
            skipped_cost += 1
            continue
        xi = int(out['exit_idx'][i])
        e = i + 1
        sl = out['entry_px'][i] - side * out['dist'][i]
        tp = out['entry_px'][i] + side * RR_RATIO * out['dist'][i]
        rows.append({
            'symbol': sd.symbol, 'side': 'LONG' if side == 1 else 'SHORT',
            'signal_bar': int(i), 'entry_bar': int(e), 'exit_bar': xi,
            'entry_ts': int(sd.ts[e]), 'exit_ts': int(sd.ts[xi]),
            'entry': float(out['entry_px'][i]), 'sl': float(sl), 'tp': float(tp),
            'exit': float(out['exit_px'][i]), 'status': S_NAME[st],
            'stop_pct': float(out['d_frac'][i] * 100),
            'r_raw': float(out['r_raw'][i]), 'cost_r': float(out['cost_r'][i]),
            'funding_r': float(out['fund_r'][i]), 'risk_usd': float(out['risk_usd'][i]),
        })
        next_free = xi
    return rows, skipped_cost, skipped_open


def apply_concurrency_cap(trades, cap):
    if cap is None or not trades:
        return trades, 0
    trades = sorted(trades, key=lambda t: t['entry_ts'])
    open_exits, kept, dropped = [], [], 0
    for t in trades:
        open_exits = [x for x in open_exits if x > t['entry_ts']]
        if len(open_exits) >= cap:
            dropped += 1
            continue
        kept.append(t)
        open_exits.append(t['exit_ts'])
    return kept, dropped


# =============================================================================
# METRICS & EVALUATION
# =============================================================================
def add_net(df, cost_mult=1.0):
    r = df['r_raw'] - cost_mult * df['cost_r'] - df['funding_r']
    return r, r * df['risk_usd']


def bootstrap_ci(x, iters=None, seed=RNG_SEED):
    iters = iters or BOOTSTRAP_ITERATIONS
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(iters, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df, win_start_ms, win_end_ms, cost_mult=1.0):
    if df is None or len(df) == 0:
        return {'n': 0}
    r, pnl = add_net(df, cost_mult)
    n = len(df)
    std = float(r.std(ddof=1)) if n > 1 else float('nan')
    mean = float(r.mean())
    d = df.assign(r=r.values, pnl=pnl.values).sort_values('exit_ts')
    eq = STARTING_BALANCE + d['pnl'].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[STARTING_BALANCE], eq]))[1:]
    dd_usd = float((eq - peak).min())
    dd_pct = float(((eq - peak) / peak).min() * 100)
    days = pd.to_datetime(d['exit_ts'], unit='ms', utc=True).dt.floor('D')
    daily = d['pnl'].groupby(days.values).sum()
    idx = pd.date_range(pd.to_datetime(win_start_ms, unit='ms', utc=True).floor('D'),
                        pd.to_datetime(win_end_ms, unit='ms', utc=True).floor('D'), freq='D')
    daily = daily.reindex(idx, fill_value=0.0)
    ret = daily / STARTING_BALANCE
    sharpe = float(ret.mean() / ret.std(ddof=1) * math.sqrt(365)) if ret.std(ddof=1) > 0 else 0.0
    gp, gl = float(d.loc[d['pnl'] > 0, 'pnl'].sum()), float(-d.loc[d['pnl'] < 0, 'pnl'].sum())
    years = max((win_end_ms - win_start_ms) / DAY / 365.25, 1e-9)
    streak = best = 0
    for v in d['pnl'].to_numpy():
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    lo, hi = bootstrap_ci(d['r'].to_numpy())
    return {
        'n': n,
        'tp_hits': int((df['status'] == 'WIN').sum()),
        'stops': int((df['status'] == 'LOSS').sum()),
        'time_stops': int((df['status'] == 'TIME_STOP').sum()),
        'profitable_pct': round(float((d['r'] > 0).mean() * 100), 1),
        'mean_r': round(mean, 3), 'median_r': round(float(d['r'].median()), 3),
        'total_r': round(float(d['r'].sum()), 1),
        'ci95_lo': round(lo, 3), 'ci95_hi': round(hi, 3),
        't_stat': round(mean / (std / math.sqrt(n)), 2) if std and std > 0 else 0.0,
        'avg_cost_r': round(float((cost_mult * df['cost_r']).mean()), 3),
        'avg_funding_r': round(float(df['funding_r'].mean()), 3),
        'pnl_usd': round(float(d['pnl'].sum()), 0),
        'return_pct': round(float(d['pnl'].sum() / STARTING_BALANCE * 100), 1),
        'ann_return_pct': round(float(d['pnl'].sum() / STARTING_BALANCE * 100 / years), 1),
        'profit_factor': round(gp / gl, 2) if gl > 0 else float('inf'),
        'max_dd_usd': round(dd_usd, 0), 'max_dd_pct': round(dd_pct, 1),
        'sharpe_ann': round(sharpe, 2), 'max_losing_streak': int(best),
    }


def random_baseline(sds, trades_df, bar_from_ts, adx_filter, iters=None, seed=RNG_SEED):
    iters = iters or MC_ITERATIONS
    rng = np.random.default_rng(seed)
    total_sum = np.zeros(iters)
    total_n = 0
    for sd in sds:
        sub = trades_df[(trades_df['symbol'] == sd.symbol) & (trades_df['entry_ts'] >= bar_from_ts)]
        for side, name in ((1, 'LONG'), (-1, 'SHORT')):
            k = int((sub['side'] == name).sum())
            if k == 0:
                continue
            out = sd.out[side]
            net = out['r_raw'] - out['cost_r'] - out['fund_r']
            mask = np.isfinite(net) & (sd.ts[np.minimum(np.arange(sd.n) + 1, sd.n - 1)] >= bar_from_ts)
            if adx_filter:
                mask &= np.isfinite(sd.adx) & (sd.adx >= ADX_MIN)
            pool = np.flatnonzero(mask)
            if len(pool) == 0:
                continue
            draw = pool[rng.integers(0, len(pool), size=(iters, k))]
            total_sum += net[draw].sum(axis=1)
            total_n += k
    if total_n == 0:
        return np.array([])
    return total_sum / total_n


def evaluate_verdict(oos, full, p_all, p_adx, oos_stress):
    g = GATES
    if oos.get('n', 0) < g['min_oos_trades']:
        return 'INCONCLUSIVE', [f"only {oos.get('n', 0)} out-of-sample trades (< {g['min_oos_trades']}); "
                                f"not enough data to accept or reject"]
    fails = []
    if oos['mean_r'] < g['min_oos_mean_r']:
        fails.append(f"OOS mean R {oos['mean_r']:+.3f} < {g['min_oos_mean_r']:+.2f}")
    if g['ci_lower_gt_zero'] and not (oos['ci95_lo'] > 0):
        fails.append(f"OOS 95% CI lower bound {oos['ci95_lo']:+.3f} is not > 0")
    for name, p in (('all-bars', p_all), ('ADX-filtered', p_adx)):
        if p is None or p > g['baseline_p_max']:
            fails.append(f"does not beat {name} random baseline (p={p if p is None else round(p, 3)})")
    if oos_stress.get('n', 0) and oos_stress['mean_r'] <= 0:
        fails.append(f"OOS mean R {oos_stress['mean_r']:+.3f} <= 0 at {g['stress_cost_mult']}x costs")
    if full.get('n', 0) and abs(full['max_dd_pct']) * STARTING_BALANCE / 100 > \
            g['max_dd_pct_of_start'] / 100 * STARTING_BALANCE:
        fails.append(f"max drawdown {full['max_dd_pct']:.1f}% of peak equity exceeds the "
                     f"{g['max_dd_pct_of_start']:.0f}% limit")
    return ('PASS' if not fails else 'FAIL'), fails


# =============================================================================
# SYMBOL LOADER PIPELINE
# =============================================================================
def analyze_symbol(symbol, d1h, d4h, d1d, fund_ts, fund_rate, test_start_ms):
    d1h, d4h, d1d = (compute_indicators(x) for x in (d1h, d4h, d1d))
    frame = build_frame(d1h, d4h, d1d)
    return SymbolData(symbol, frame, fund_ts, fund_rate, test_start_ms)


def load_symbol(symbol, start_ms, end_ms, use_cache):
    inst = inst_id_of(symbol)
    base = symbol.split('/')[0]
    print(f"\n=== {symbol} ({inst}) ===")
    dfs = {}
    for tf in ('1d', '4h', '1h'):
        step = TF[tf][1]
        warm = int(MIN_BARS * step * 1.15)
        print(f"    Fetching {tf} from Binance Vision...")
        df = fetch_candles_binance_vision(symbol, tf, start_ms - warm, end_ms)
        if df.empty or len(df) < MIN_BARS * 3:
            print(f"    Binance Vision sparse for {tf}, falling back to OKX")
            df = fetch_candles(inst, tf, start_ms - warm, end_ms, use_cache)
        audit_candles(df, tf, base, start_ms - warm, end_ms)
        dfs[tf] = df
    h = dfs['1h']
    last_ts = int(h['ts'].iloc[-1])
    if last_ts < end_ms - 30 * DAY:
        print(f"    ⚠️  Binance Vision ends {(end_ms - last_ts)/DAY:.1f}d before now (expected lag)")

    # Funding: Binance Vision for older, OKX for recent
    bnv = fetch_funding_binance_vision(base, start_ms, end_ms)
    try:
        okx_f = fetch_funding_okx(inst, start_ms)
    except (DataError, FatalAPIError):
        okx_f = []
    okx_rows = [(t, r, 'okx') for t, r in okx_f if start_ms <= t <= end_ms]
    bn_rows = [(t, r, 'binance') for t, r in bnv if start_ms <= t <= end_ms]
    real = sorted(bn_rows + okx_rows, key=lambda x: x[0])
    fts, frate, fsrc = fill_funding_gaps(real, start_ms, end_ms, FALLBACK_FUNDING_8H)
    n_real = sum(1 for s in fsrc if s != 'assumed')
    print(f"    funding: {len(fsrc)} points | okx {fsrc.count('okx')} | binance {fsrc.count('binance')} "
          f"| assumed {fsrc.count('assumed')} ({fsrc.count('assumed')/max(len(fsrc),1)*100:.0f}%)")
    sd = analyze_symbol(symbol, dfs['1h'], dfs['4h'], dfs['1d'], fts, frate, start_ms)
    sd.funding_real_pct = n_real / max(len(fsrc), 1) * 100
    return sd


# =============================================================================
# REPORTING
# =============================================================================
def fmt_line(label, m):
    if not m or m.get('n', 0) == 0:
        return f"{label:<14} no trades"
    return (f"{label:<14} n={m['n']:4d} | mean {m['mean_r']:+.3f}R [{m['ci95_lo']:+.2f},{m['ci95_hi']:+.2f}] "
            f"| PF {m['profit_factor']:.2f} | ${m['pnl_usd']:+,.0f} ({m['return_pct']:+.1f}%) "
            f"| DD {m['max_dd_pct']:.1f}% | Sharpe {m['sharpe_ann']:.2f}")


def breakdown(df, key_fn, label):
    r, pnl = add_net(df)
    d = df.assign(r=r.values, pnl=pnl.values, k=key_fn(df))
    print(f"\n--- BY {label} ---")
    for k, g in d.groupby('k', sort=True, observed=True):
        print(f"{str(k):<14} n={len(g):4d} | mean {g['r'].mean():+.3f}R | avg cost {g['cost_r'].mean():.2f}R "
              f"| TP-hit {(g['status'] == 'WIN').mean() * 100:4.1f}% | ${g['pnl'].sum():+,.0f}")


def report(sds, trades, args, start_ms, end_ms, split_ms, skipped):
    fmt = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime('%Y-%m-%d')
    print('\n' + '=' * 78)
    print('                    BACKTEST REPORT (v3, 4-Year Binance Vision Run)')
    print('=' * 78)
    print(f"Test window : {fmt(start_ms)} -> {fmt(end_ms)}  ({(end_ms - start_ms) / DAY:.0f} days)")
    print(f"In-sample   : {fmt(start_ms)} -> {fmt(split_ms)} | Out-of-sample: {fmt(split_ms)} -> {fmt(end_ms)}")
    print(f"Symbols     : {', '.join(sd.symbol.split('/')[0] for sd in sds)}")
    print(f"Sizing      : ${STARTING_BALANCE * RISK_PER_TRADE:.0f} risk/trade (fixed, non-compounding), "
          f"leverage cap {LEVERAGE_CAP}x, max concurrent: {MAX_CONCURRENT or 'none'}")
    print(f"Skipped     : {skipped['cost']} signals by cost filter, {skipped['open']} still open at data end, "
          f"{skipped['cap']} by concurrency cap")

    if trades.empty:
        print('\nNO TRADES. Nothing to evaluate.')
        return None

    full = summarize(trades, start_ms, end_ms)
    ins = summarize(trades[trades['entry_ts'] < split_ms], start_ms, split_ms)
    oos = summarize(trades[trades['entry_ts'] >= split_ms], split_ms, end_ms)
    stress = summarize(trades[trades['entry_ts'] >= split_ms], split_ms, end_ms, GATES['stress_cost_mult'])

    print('\n--- RESULTS (net of fees, slippage, funding) ---')
    print(fmt_line('ALL', full))
    print(fmt_line('IN-SAMPLE', ins))
    print(fmt_line('OUT-OF-SAMPLE', oos))
    print(f"   exits: {full['tp_hits']} TP / {full['stops']} stop / {full['time_stops']} time-stop | "
          f"profitable {full['profitable_pct']}% | avg cost {full['avg_cost_r']:.2f}R | "
          f"avg funding {full['avg_funding_r']:+.3f}R | longest losing streak {full['max_losing_streak']}")
    print(f"   t-stat {full['t_stat']} | ann. return {full['ann_return_pct']}%/yr on starting balance")

    print('\n--- PER SYMBOL ---')
    for sd in sds:
        sub = trades[trades['symbol'] == sd.symbol]
        print(fmt_line(sd.symbol.split('/')[0], summarize(sub, start_ms, end_ms)) +
              f" | fund-real {sd.funding_real_pct:.0f}%")

    breakdown(trades, lambda d: d['side'], 'DIRECTION')
    breakdown(trades, lambda d: pd.cut(d['stop_pct'], [0, 0.5, 1.0, 2.0, 100],
                                       labels=['<0.5%', '0.5-1%', '1-2%', '>2%']), 'STOP DISTANCE')

    print('\n--- RANDOM-ENTRY BASELINE (Monte Carlo) ---')
    p = {}
    for tag, adx_f in (('all-bars', False), ('ADX-filtered', True)):
        for scope, frm in (('full', start_ms), ('OOS', split_ms)):
            base = random_baseline(sds, trades, frm, adx_f)
            sub = trades[trades['entry_ts'] >= frm]
            if len(base) == 0 or sub.empty:
                continue
            actual = float(add_net(sub)[0].mean())
            pv = float((np.sum(base >= actual) + 1) / (len(base) + 1))
            p[(tag, scope)] = pv
            print(f"  {tag:<13} {scope:<4}: actual {actual:+.3f}R vs random mean {base.mean():+.3f}R | p = {pv:.3f}")

    verdict, reasons = evaluate_verdict(oos, full, p.get(('all-bars', 'OOS')),
                                        p.get(('ADX-filtered', 'OOS')), stress)
    print('\n' + '=' * 78)
    print(f"VERDICT (pre-registered gates): {verdict}")
    for r_ in reasons:
        print(f"  - {r_}")
    print('=' * 78)
    return {'full': full, 'in_sample': ins, 'oos': oos, 'oos_stress': stress,
            'verdict': verdict, 'reasons': reasons}


# =============================================================================
# MAIN & SELF-TEST
# =============================================================================
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    return str(o)


def run(args):
    global CACHE_DIR
    CACHE_DIR = args.cache_dir
    end_ms = int(time.time() * 1000) // HOUR * HOUR
    start_ms = end_ms - int(args.days * DAY)
    split_ms = start_ms + int((end_ms - start_ms) * TRAIN_RATIO)
    symbols = SYMBOLS
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        symbols = [s for s in SYMBOLS if s.split('/')[0] in wanted]
    print(f"Backtest v3 | {len(symbols)} symbols | {args.days} days | cache={'off' if args.no_cache else 'on'}")

    sds, failures = [], []
    for sym in symbols:
        try:
            sds.append(load_symbol(sym, start_ms, end_ms, not args.no_cache))
        except (DataError, FatalAPIError) as e:
            print(f"  !! {sym} skipped: {e}")
            failures.append(sym)
    if not sds:
        print("No usable data. Aborting.")
        return 1

    rows, skipped_cost, skipped_open = [], 0, 0
    for sd in sds:
        r, sc, so = select_trades(sd, args.max_cost_r if args.max_cost_r is not None else MAX_COST_R)
        rows.extend(r)
        skipped_cost += sc
        skipped_open += so
    rows, dropped = apply_concurrency_cap(rows, MAX_CONCURRENT)
    trades = pd.DataFrame(rows)
    res = report(sds, trades, args, start_ms, end_ms, split_ms,
                 {'cost': skipped_cost, 'open': skipped_open, 'cap': dropped})

    os.makedirs(args.out_dir, exist_ok=True)
    if not trades.empty:
        out = trades.copy()
        out['entry_time'] = pd.to_datetime(out['entry_ts'], unit='ms', utc=True)
        out['exit_time'] = pd.to_datetime(out['exit_ts'], unit='ms', utc=True)
        out['r_net'], out['pnl_usd'] = add_net(out)
        out.to_csv(os.path.join(args.out_dir, 'backtest_trades.csv'), index=False)
    with open(os.path.join(args.out_dir, 'backtest_report.json'), 'w') as f:
        json.dump({'generated': datetime.now(timezone.utc).isoformat(),
                   'window_days': args.days, 'symbols': [s.symbol for s in sds],
                   'failed_symbols': failures, 'results': res},
                  f, indent=2, default=_json_default)
    print(f"\nSaved {args.out_dir}/backtest_report.json and backtest_trades.csv")
    return 0


def main():
    ap = argparse.ArgumentParser(description='OKX/Binance perp swing-strategy backtester (v3)')
    ap.add_argument('--selftest', action='store_true', help='run offline logic tests and exit')
    ap.add_argument('--days', type=int, default=BACKTEST_DAYS)
    ap.add_argument('--symbols', nargs='*', help='subset by base, e.g. BTC ETH')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--cache-dir', default=CACHE_DIR)
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--max-cost-r', type=float, default=None)
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == '__main__':
    main()
