import requests
import pandas as pd
from datetime import datetime, timedelta

# API key and base URL
api_key = "ul3lmuulialb30wc5bvqhe3fdlnmhfu2xd5k4v9i"
base_url = "https://stocknewsapi.com/api/v1/category?section=alltickers&items=50&page={page}&token=" + api_key


# Function to generate 3-month date ranges
def generate_date_ranges(start_date, end_date):
    current_start_date = start_date
    while current_start_date < end_date:
        current_end_date = min(current_start_date + timedelta(days=90), end_date)
        yield current_start_date, current_end_date
        current_start_date = current_end_date + timedelta(days=1)


# Function to fetch news for a specific date range
def fetch_news(page, start_date, end_date):
    formatted_start_date = start_date.strftime("%m%d%Y")
    formatted_end_date = end_date.strftime("%m%d%Y")
    url = base_url.format(page=page) + f"&date={formatted_start_date}-{formatted_end_date}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Assuming the API returns JSON data
    else:
        return None


# Main function to aggregate news
def aggregate_news(start_date, end_date):
    all_news = []
    for start, end in generate_date_ranges(start_date, end_date):
        page = 1
        while True:
            news_data = fetch_news(page, start, end)
            if news_data and news_data['data']:
                all_news.extend(news_data['data'])
                page += 1
                print(f"Fetching page: {page} for date range {start} to {end}")
            else:
                break
    return all_news


if __name__ == "__main__":
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 5, 13)

    news_data = aggregate_news(start_date, end_date)

    # Creating a DataFrame with all aggregated news
    df = pd.DataFrame(news_data)

    # Saving the DataFrame as an Excel file
    df.to_excel('aggregated_news.xlsx', index=False)
    print("DataFrame has been saved as 'aggregated_news.xlsx'.")
