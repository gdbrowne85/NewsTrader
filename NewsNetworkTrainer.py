import pandas as pd
import ast

# Load the dataset
df = pd.read_excel('aggregated_news.xlsx')

# Convert the 'date' column to datetime objects with UTC as the timezone
df['date'] = pd.to_datetime(df['date'], utc=True)

# Convert these datetimes to Eastern Time
df['date'] = df['date'].dt.tz_convert('America/New_York')

# Drop rows based on 'type' and weekends
df = df[~df['type'].str.contains('Video')]

# Drop the unnecessary columns
df.drop(columns=['news_url', 'image_url', 'type', 'source_name'], inplace=True)

# Assume that 'tickers' is a string representation of a list
# Convert it to an actual list using ast.literal_eval
df['tickers'] = df['tickers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Keep rows where the "tickers" list has exactly one element
df = df[df['tickers'].apply(lambda x: len(x) == 1)]

# Extract the single ticker from each list
df['tickers'] = df['tickers'].apply(lambda x: x[0] if isinstance(x, list) else x)

# Remove timezone information from 'date'
df['date'] = df['date'].dt.tz_localize(None)

# Save the DataFrame to Excel without timezone information
df.to_excel('pre_processed_news.xlsx', index=False)
