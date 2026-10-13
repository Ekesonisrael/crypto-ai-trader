import ccxt
import pandas as pd
import os
from datetime import datetime
import json

bybit = ccxt.bybit({'timeout': 30000, 'options': {'defaultType': 'swap', 'fetchMarkets': ['linear']}})
JOURNAL_FILE = 'trade_journal.csv'
SYMBOL = 'BTC/USDT:USDT'

if not os.path.isfile(JOURNAL_FILE):
    print("No trade journal found. Run fetch_prices.py first.")
else:
    print("Checking trade journal for unresolved trades...\n")
    df = pd.read_csv(JOURNAL_FILE)

    ohlcv = bybit.fetch_ohlcv(SYMBOL, '5m', limit=100)
    market_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'], unit='ms')

    updated = False
    for index, row in df.iterrows():
        if row['Status'] == 'PENDING':
            trade_time = pd.to_datetime(row['Date'])
            future_candles = market_df[market_df['timestamp'] >= trade_time]
            
            if future_candles.empty:
                continue 
                
            highest_price = future_candles['high'].max()
            lowest_price = future_candles['low'].min()
            tp = float(row['Take_Profit'])
            sl = float(row['Stop_Loss'])
            
            result = None
            if row['Direction'] == 'LONG':
                if lowest_price <= sl: result = 'LOSS'
                elif highest_price >= tp: result = 'WIN'
            elif row['Direction'] == 'SHORT':
                if highest_price >= sl: result = 'LOSS'
                elif lowest_price <= tp: result = 'WIN'
                    
            if result:
                df.at[index, 'Status'] = result
                updated = True
                print(f"Trade from {row['Date']} resolved as: {result}")

    if updated:
        df.to_csv(JOURNAL_FILE, index=False)
        print("\nJournal updated successfully!")
    else:
        print("No trades resolved yet. Still waiting for price to hit TP or SL.")

    print("\n--- PERFORMANCE STATS ---")
    wins = len(df[df['Status'] == 'WIN'])
    losses = len(df[df['Status'] == 'LOSS'])
    pending = len(df[df['Status'] == 'PENDING'])
    total_resolved = wins + losses

    print(f"Total Trades Logged: {len(df)}")
    print(f"Wins: {wins} | Losses: {losses} | Pending: {pending}")

    if total_resolved > 0:
        win_rate = (wins / total_resolved) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    else:
        print("Win Rate: N/A (No trades resolved yet)")

    # Save Stats for Chief Agent
    if total_resolved > 0:
        win_rate_val = (wins / total_resolved) * 100
    else:
        win_rate_val = 50.0

    stats_data = {
        "total_trades": int(len(df)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate_val, 2)
    }
    with open('bot_stats.json', 'w') as f:
        json.dump(stats_data, f, indent=4)
    print("\n✅ Stats saved to bot_stats.json for the Chief Agent to read.")
