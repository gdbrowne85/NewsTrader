import pandas as pd
import requests
from datetime import datetime, timedelta
import time

API_KEY = '14ee93fb9b974790a5d784ec5b8622ad'  # TwelveData API Key
endpoint = 'https://api.twelvedata.com/time_series'

# Load the dataset
news = pd.read_excel("pre_processed_news.xlsx")

# Add new columns for the required price information
"""
news['open'] = None
news['five_minutes_prior_price'] = None
news['concurrent_price'] = None
news['close'] = None
news['low'] = None
news['high'] = None
"""
news['previous_close'] = None
news['next_close'] = None
news['sentiment_score'] = None

total_rows = len(news)

# Function to format the time remaining
def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}"

start_time = time.time()

# Iterate over the DataFrame rows to make API requests
for index, row in news.iterrows():
    #time.sleep(0.1)
    print(f"Processing row {index + 1} of {total_rows}")
    symbol = row['tickers']
    timestamp = pd.Timestamp(row['date'])
    date = timestamp.date()
    """
    # Fetch the day's high, low, open, and close prices
    params_daily = {
        'symbol': symbol,
        'interval': '1day',
        'apikey': API_KEY,
        'date': date
    }
    response_daily = requests.get(endpoint, params=params_daily)
    daily_data = response_daily.json()

    # Update DataFrame with daily data
    if 'values' in daily_data and daily_data['values']:
        day_values = daily_data['values'][0]
        news.at[index, 'open'] = day_values['open']
        news.at[index, 'close'] = day_values['close']
        news.at[index, 'high'] = day_values['high']
        news.at[index, 'low'] = day_values['low']
        news.at[index, 'sentiment_score'] = 100 * round(
            (float(day_values['close']) - float(day_values['open'])) / float(day_values['open']), 3)
    """
    # Fetch the previous trading day's close price
    previous_trading_day = timestamp - timedelta(days=1)
    while previous_trading_day.weekday() > 4:  # Skip weekends
        previous_trading_day -= timedelta(days=1)

    params_previous = {
        'symbol': symbol,
        'interval': '1day',
        'apikey': API_KEY,
        'date': previous_trading_day
    }
    response_previous = requests.get(endpoint, params=params_previous)
    previous_data = response_previous.json()

    if 'values' in previous_data and previous_data['values']:
        previous_close = previous_data['values'][0]['close']
        news.at[index, 'previous_close'] = previous_close

    # Fetch the next trading day's close price
    next_trading_day = timestamp + timedelta(days=1)
    while next_trading_day.weekday() > 4:  # Skip weekends
        next_trading_day += timedelta(days=1)


    params_next = {
        'symbol': symbol,
        'interval': '1day',
        'apikey': API_KEY,
        'date': next_trading_day
    }
    response_next = requests.get(endpoint, params=params_next)
    next_data = response_next.json()


    if 'values' in next_data and next_data['values']:
        next_close = next_data['values'][0]['close']
        news.at[index, 'next_close'] = next_close


        # Calculate sentiment_score
        if previous_close and next_close:
            sentiment_score = ((float(next_close) - float(previous_close)) / float(previous_close)) * 100
            news.at[index, 'sentiment_score'] = round(sentiment_score, 1)


    # Calculate progress
    elapsed_time = time.time() - start_time
    rows_processed = index + 1
    percent_complete = (rows_processed / total_rows) * 100
    time_per_row = elapsed_time / rows_processed
    time_remaining = time_per_row * (total_rows - rows_processed)

    print(f"Progress: {percent_complete:.2f}% complete. Estimated time remaining: {format_time(time_remaining)}")

# Save the updated DataFrame to a new Excel file
news.to_excel("updated_pre_processed_news.xlsx", index=False)
print("Data processing complete. Updated file saved as 'updated_pre_processed_news.xlsx'.")
