from bs4 import BeautifulSoup
import matplotlib
import requests
import pandas as pd
from datetime import datetime, timedelta
matplotlib.use('TkAgg')

def fetch_yahoo_finance_data(url: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch data from {url}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find("table", {"data-test": "historical-prices"})
    
    if table is None:
        print(f"No historical data found for URL {url}")
        return None

    headers = []
    for header in table.findAll("th"):
        headers.append(header.text)
    
    rows = table.findAll("tr")
    data = []
    
    for row in rows[1:]:
        cells = row.findAll("td")
        if len(cells) != len(headers):
            continue
        row_data = {}
        for i in range(len(cells)):
            row_data[headers[i]] = cells[i].text
        data.append(row_data)
        
    df = pd.DataFrame(data)
    df = df.dropna()

    for col in ["Open*", "High", "Low", "Close*", "Adj Close**"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
        
    df['Date'] = pd.to_datetime(df['Date'])
    df['Daily Return'] = (df['Adj Close**'] / df['Adj Close**'].shift(1)) - 1

    last_date = df['Date'].max()
    one_year_ago = pd.Timestamp(last_date.year - 1, last_date.month, last_date.day)
    df_filtered = df[df['Date'] > one_year_ago]

    return df_filtered

def fetch_binance_data(symbol='BTCUSDT', interval='1d', limit=365):
    url = f"https://api.binance.com/api/v3/klines"
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (365 * 24 * 60 * 60 * 1000)
    
    params = {'symbol': symbol, 'interval': interval, 'limit': limit, 'startTime': start_time, 'endTime': end_time}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        raw_data = response.json()
        df = pd.DataFrame(raw_data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        df['Daily Return'] = (df['Close'] / df['Close'].shift(1)) - 1

        return df

    else:
        print("Failed to fetch data from Binance")
        return None
def fetch_binance_data_formatted(symbol='BTC_USDT', interval='1h', limit=1):
    url = "https://api.binance.com/api/v3/klines"

    end_time = int(datetime.now().timestamp() * 1000)

    interval_duration = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '6h': 360,
                         '8h': 480, '12h': 720, '1d': 1440}
    minutes_to_subtract = interval_duration[interval] * limit
    start_time = end_time - (minutes_to_subtract * 60 * 1000)

    symbolreq = symbol.replace('_', '')
    params = {'symbol': symbolreq, 'interval': interval, 'limit': limit, 'startTime': start_time, 'endTime': end_time}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        raw_data = response.json()
        formatted_data = []

        for data in raw_data:
            formatted_candle = {
                'pair': symbol,
                'date': pd.to_datetime(data[0], unit='ms'),
                'open': float(data[1]),
                'high': float(data[2]),
                'low': float(data[3]),
                'close': float(data[4]),
                'volume': float(data[5])
            }
            formatted_data.append(formatted_candle)

        return formatted_data
    else:
        print("Failed to fetch data from Binance")
        return None