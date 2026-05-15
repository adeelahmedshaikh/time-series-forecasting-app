# news_sentiment.py
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import streamlit as st

class NewsSentimentAnalyzer:
    def __init__(self, news_api_key=None):
        self.news_api_key = news_api_key
        # Free fallback without API key
        self.use_free_mode = news_api_key is None
    
    def fetch_news_free(self, ticker):
        """Free method: Use Google News RSS (no API key needed)"""
        import feedparser
        
        news_list = []
        try:
            # Google News RSS feed
            rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:5]:
                # Analyze sentiment
                blob = TextBlob(entry.title)
                sentiment_score = blob.sentiment.polarity  # -1 to 1
                
                news_list.append({
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.published,
                    'sentiment_score': sentiment_score,
                    'sentiment_label': 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral',
                    'source': entry.source.title if hasattr(entry, 'source') else 'Google News'
                })
        except Exception as e:
            st.warning(f"Could not fetch news: {e}")
        
        return news_list
    
    def fetch_news_api(self, ticker):
        """Using NewsAPI (requires API key)"""
        if not self.news_api_key:
            return self.fetch_news_free(ticker)
        
        try:
            # Calculate date from 7 days ago
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f'{ticker} stock',
                'from': from_date,
                'sortBy': 'relevancy',
                'pageSize': 10,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            news_list = []
            if data.get('status') == 'ok':
                for article in data['articles']:
                    # Combine title + description for better sentiment analysis
                    text = f"{article['title']} {article.get('description', '')}"
                    blob = TextBlob(text)
                    sentiment_score = blob.sentiment.polarity
                    
                    news_list.append({
                        'title': article['title'],
                        'link': article['url'],
                        'published': article['publishedAt'],
                        'sentiment_score': sentiment_score,
                        'sentiment_label': 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral',
                        'source': article['source']['name']
                    })
            return news_list[:10]
        except Exception as e:
            st.error(f"News API error: {e}")
            return []
    
    def get_sentiment_summary(self, news_list):
        """Calculate overall sentiment from news"""
        if not news_list:
            return None
        
        scores = [item['sentiment_score'] for item in news_list]
        avg_score = sum(scores) / len(scores)
        
        positive_count = sum(1 for item in news_list if item['sentiment_label'] == 'Positive')
        negative_count = sum(1 for item in news_list if item['sentiment_label'] == 'Negative')
        neutral_count = sum(1 for item in news_list if item['sentiment_label'] == 'Neutral')
        
        sentiment_multiplier = 1 + (avg_score * 0.1)  # -10% to +10% impact
        
        return {
            'avg_sentiment_score': avg_score,
            'sentiment_multiplier': sentiment_multiplier,
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'overall': 'Positive' if avg_score > 0.1 else 'Negative' if avg_score < -0.1 else 'Neutral'
        }
    
    def display_news(self, news_list, max_display=5):
        """Display news in Streamlit UI"""
        if not news_list:
            st.info("No recent news found for this ticker.")
            return
        
        for i, article in enumerate(news_list[:max_display]):
            sentiment_icon = "🟢" if article['sentiment_label'] == 'Positive' else "🔴" if article['sentiment_label'] == 'Negative' else "⚪"
            sentiment_color = "green" if article['sentiment_label'] == 'Positive' else "red" if article['sentiment_label'] == 'Negative' else "gray"
            
            st.markdown(f"""
            <div style='padding: 10px; border-left: 3px solid {sentiment_color}; margin-bottom: 10px;'>
                <strong>{sentiment_icon} {article['title']}</strong><br>
                <small>📰 {article['source']} | Sentiment: {article['sentiment_label']} ({article['sentiment_score']:.2f})</small>
            </div>
            """, unsafe_allow_html=True)