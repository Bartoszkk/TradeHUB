import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from bot import plot_queue
import mplfinance as mpf


def generate_graph(plot_data, symbol="SOL_USDT"):
    try:
        candle_data = plot_data.get('candle_data', [])
        if not candle_data:
            raise ValueError("Empty candle data")

        df = pd.DataFrame(candle_data)
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.set_index('date', inplace=True)

        usdt_values = plot_data.get('usdt_values', [])
        if not usdt_values:
            raise ValueError("Empty USDT values")

        btc_values = plot_data.get('btc_values', [])
        if not btc_values:
            raise ValueError("Empty BTC values")

        # Convert to DataFrame if 'time' key exists
        if all('time' in val for val in usdt_values):
            usdt_df = pd.DataFrame(usdt_values)
            usdt_df['date'] = pd.to_datetime(usdt_df['time'], unit='s')
            usdt_df.set_index('date', inplace=True)
            usdt_df.sort_index(inplace=True)
        else:
            raise KeyError("'time' key missing in usdt_values")

        if all('time' in val for val in btc_values):
            btc_df = pd.DataFrame(btc_values)
            btc_df['date'] = pd.to_datetime(btc_df['time'], unit='s')
            btc_df.set_index('date', inplace=True)
            btc_df.sort_index(inplace=True)
        else:
            raise KeyError("'time' key missing in btc_values")
        usdt_line = mpf.make_addplot(usdt_df['value'], type='line', color='orange', panel=2, ylabel=symbol.split('_')[1])
        btc_line = mpf.make_addplot(btc_df['value'], type='line', color='purple', panel=3, ylabel=symbol.split('_')[0])

        additional_plots = [usdt_line, btc_line]

        mpf.plot(df, type='candle', style='charles',
                 title=f'Candlestick chart for {symbol}',
                 ylabel='Price',
                 addplot=additional_plots,
                 volume=True,
                 volume_panel=1,
                 panel_ratios=(6, 3, 2, 2),
                 figratio=(12, 6),
                 figscale=1.5,
                 savefig='output.png')
        print("Graph saved successfully.")
    except Exception as e:
        print(f"Error in generating graph: {e}")

