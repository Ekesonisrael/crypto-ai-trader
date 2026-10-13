import ccxt
import pandas as pd
import os
import json
from datetime import datetime

bybit = ccxt.okx({'timeout': 30000, 'options': {'defaultType': 'swap'}})
JOURNAL_FILE = 'trade_journal.csv'
STATS_FILE = 'bot_stats.json'
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
            
            if future_candles.empty: continue 
                
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

    # Calculate Stats and Update Balance
    wins = len(df[df['Status'] == 'WIN'])
    losses = len(df[df['Status'] == 'LOSS'])
    pending = len(df[df['Status'] == 'PENDING'])
    total_resolved = wins + losses

    # Load previous balance
    current_balance = 10000
    if os.path.isfile(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                old_stats = json.load(f)
                current_balance = old_stats.get('current_balance', 10000)
        except: pass

    # Calculate P&L for newly resolved trades
    # Note: This is a simplified calculation based on 1% risk per trade
    for index, row in df.iterrows():
        if row['Status'] == 'WIN' and 'processed' not in row:
            # Assume 1% risk, 2% reward
            current_balance += (current_balance * 0.02) 
            df.at[index, 'processed'] = True
        elif row['Status'] == 'LOSS' and 'processed' not in row:
            current_balance -= (current_balance * 0.01)
            df.at[index, 'processed'] = True

    df.to_csv(JOURNAL_FILE, index=False)

    if total_resolved > 0:
        win_rate = (wins / total_resolved) * 100
    else:
        win_rate = 50.0

    print(f"\n--- PERFORMANCE STATS ---")
    print(f"Total Trades Logged: {len(df)}")
    print(f"Wins: {wins} | Losses: {losses} | Pending: {pending}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Simulated Balance: ${current_balance:.2f}")

    # Save Stats for Chief Agent
    stats_data = {
        "total_trades": int(len(df)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate, 2),
        "current_balance": round(current_balance, 2)
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats_data, f, indent=4)
    print("\n✅ Stats saved to bot_stats.json for the Chief Agent to read.")
