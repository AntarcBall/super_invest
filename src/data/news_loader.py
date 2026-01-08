from gnews import GNews
from typing import List, Dict
import pandas as pd

def get_news(ticker: str, start_date: tuple, end_date: tuple) -> List[Dict]:
    """
    Fetches news articles for a given ticker from Google News.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (tuple): The start date as a tuple (YYYY, MM, DD).
        end_date (tuple): The end date as a tuple (YYYY, MM, DD).

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a news article.
                     Returns an empty list if no news is found or an error occurs.
    """
    try:
        google_news = GNews(language='en', country='US')
        google_news.start_date = start_date
        google_news.end_date = end_date
        
        # GNews works best with company names, so we'll search for the ticker
        # as a keyword. This is a common practice.
        news_list = google_news.get_news(f'{ticker} stock')

        if not news_list:
            print(f"Warning: No news found for ticker '{ticker}' for the given date range.")
            return []

        # Format the articles for consistency
        formatted_news = [
            {
                'published_date': article['published date'],
                'title': article['title'],
                'description': article['description'],
                'url': article['url']
            }
            for article in news_list
        ]
        
        print(f"Successfully fetched {len(formatted_news)} news articles for {ticker}.")
        return formatted_news

    except Exception as e:
        print(f"An error occurred while fetching news for ticker '{ticker}': {e}")
        return []

if __name__ == '__main__':
    # Example usage:
    ticker = 'TSLA'
    start = (2024, 5, 1)
    end = (2024, 5, 10)
    
    tsla_news = get_news(ticker, start, end)
    
    if tsla_news:
        print(f"\nFound {len(tsla_news)} articles for {ticker}. Showing first 3:")
        for i, article in enumerate(tsla_news[:3]):
            print(f"\n--- Article {i+1} ---")
            print(f"Date: {article['published_date']}")
            print(f"Title: {article['title']}")
            print(f"Description: {article['description']}")
            print(f"URL: {article['url']}")
