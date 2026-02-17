# -*- coding: utf-8 -*-
"""
Data Ingestion Module for Hate Speech Detection
Supports: Twitter, Facebook, Reddit, Web Scraping, CSV Files
"""

import os
import csv
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
from urllib.parse import urljoin, urlparse
import pandas as pd

# Social Media API Libraries
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

# Web Scraping
try:
    from bs4 import BeautifulSoup
    import requests
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False

# Facebook Graph API (using requests)
try:
    import facebook
    FACEBOOK_AVAILABLE = True
except ImportError:
    FACEBOOK_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataIngestionManager:
    """Manages data ingestion from multiple sources"""
    
    def __init__(self):
        self.processed_data = []
        self.setup_api_credentials()
    
    def setup_api_credentials(self):
        """Setup API credentials from environment variables"""
        self.twitter_credentials = {
            'bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
            'consumer_key': os.getenv('TWITTER_CONSUMER_KEY'),
            'consumer_secret': os.getenv('TWITTER_CONSUMER_SECRET'),
            'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
            'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        }
        
        self.reddit_credentials = {
            'client_id': os.getenv('REDDIT_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'user_agent': os.getenv('REDDIT_USER_AGENT', 'HateSpeechDetector/1.0')
        }
        
        self.facebook_credentials = {
            'access_token': os.getenv('FACEBOOK_ACCESS_TOKEN'),
            'app_id': os.getenv('FACEBOOK_APP_ID'),
            'app_secret': os.getenv('FACEBOOK_APP_SECRET')
        }
    
    def ingest_from_twitter(self, query: str, count: int = 100, language: str = 'en') -> List[Dict[str, Any]]:
        """Ingest tweets from Twitter API"""
        if not TWITTER_AVAILABLE:
            logger.error("Twitter library not available. Install tweepy: pip install tweepy")
            return []
        
        try:
            # Use Twitter API v2 with Bearer Token
            client = tweepy.Client(
                bearer_token=self.twitter_credentials['bearer_token'],
                consumer_key=self.twitter_credentials['consumer_key'],
                consumer_secret=self.twitter_credentials['consumer_secret'],
                access_token=self.twitter_credentials['access_token'],
                access_token_secret=self.twitter_credentials['access_token_secret']
            )
            
            # Search for tweets
            tweets = client.search_recent_tweets(
                query=query,
                max_results=min(count, 100),  # Twitter API limit
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang'],
                lang=language
            )
            
            ingested_data = []
            if tweets.data:
                for tweet in tweets.data:
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'author_id': tweet.author_id,
                        'created_at': tweet.created_at,
                        'language': tweet.lang,
                        'source': 'twitter',
                        'metrics': tweet.public_metrics,
                        'raw_data': tweet.data if hasattr(tweet, 'data') else {}
                    }
                    ingested_data.append(tweet_data)
            
            logger.info(f"Successfully ingested {len(ingested_data)} tweets from Twitter")
            return ingested_data
            
        except Exception as e:
            logger.error(f"Twitter ingestion failed: {str(e)}")
            return []
    
    def ingest_from_reddit(self, subreddit: str, post_type: str = 'hot', limit: int = 100) -> List[Dict[str, Any]]:
        """Ingest posts and comments from Reddit"""
        if not REDDIT_AVAILABLE:
            logger.error("Reddit library not available. Install praw: pip install praw")
            return []
        
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_credentials['client_id'],
                client_secret=self.reddit_credentials['client_secret'],
                user_agent=self.reddit_credentials['user_agent']
            )
            
            ingested_data = []
            
            # Get subreddit
            sub = reddit.subreddit(subreddit)
            
            # Get posts based on type
            if post_type == 'hot':
                posts = sub.hot(limit=limit)
            elif post_type == 'new':
                posts = sub.new(limit=limit)
            elif post_type == 'top':
                posts = sub.top(limit=limit)
            else:
                posts = sub.hot(limit=limit)
            
            for post in posts:
                # Add post data
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'author': str(post.author) if post.author else '[deleted]',
                    'created_at': datetime.fromtimestamp(post.created_utc),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'subreddit': subreddit,
                    'source': 'reddit_post',
                    'url': post.url
                }
                ingested_data.append(post_data)
                
                # Add top comments
                post.comments.replace_more(limit=0)
                for comment in post.comments[:10]:  # Top 10 comments
                    comment_data = {
                        'id': comment.id,
                        'text': comment.body,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'created_at': datetime.fromtimestamp(comment.created_utc),
                        'score': comment.score,
                        'subreddit': subreddit,
                        'source': 'reddit_comment',
                        'parent_post_id': post.id
                    }
                    ingested_data.append(comment_data)
            
            logger.info(f"Successfully ingested {len(ingested_data)} items from Reddit r/{subreddit}")
            return ingested_data
            
        except Exception as e:
            logger.error(f"Reddit ingestion failed: {str(e)}")
            return []
    
    def ingest_from_facebook(self, page_id: str, post_limit: int = 50) -> List[Dict[str, Any]]:
        """Ingest posts from Facebook Graph API"""
        if not self.facebook_credentials['access_token']:
            logger.error("Facebook access token not configured")
            return []
        
        try:
            base_url = "https://graph.facebook.com/v18.0"
            access_token = self.facebook_credentials['access_token']
            
            ingested_data = []
            
            # Get page posts
            posts_url = f"{base_url}/{page_id}/posts"
            params = {
                'access_token': access_token,
                'limit': post_limit,
                'fields': 'id,message,created_time,likes.summary(true),comments.summary(true),shares'
            }
            
            response = requests.get(posts_url, params=params)
            response.raise_for_status()
            
            posts_data = response.json()
            
            for post in posts_data.get('data', []):
                post_data = {
                    'id': post['id'],
                    'text': post.get('message', ''),
                    'created_at': datetime.fromisoformat(post['created_at'].replace('Z', '+00:00')),
                    'likes_count': post.get('likes', {}).get('summary', {}).get('total_count', 0),
                    'comments_count': post.get('comments', {}).get('summary', {}).get('total_count', 0),
                    'shares_count': post.get('shares', {}).get('count', 0),
                    'source': 'facebook',
                    'page_id': page_id
                }
                ingested_data.append(post_data)
            
            logger.info(f"Successfully ingested {len(ingested_data)} posts from Facebook page {page_id}")
            return ingested_data
            
        except Exception as e:
            logger.error(f"Facebook ingestion failed: {str(e)}")
            return []
    
    def ingest_from_web_scraping(self, urls: List[str], selector: str = None) -> List[Dict[str, Any]]:
        """Ingest content from websites using web scraping"""
        if not WEB_SCRAPING_AVAILABLE:
            logger.error("Web scraping libraries not available. Install beautifulsoup4, requests, selenium")
            return []
        
        ingested_data = []
        
        for url in urls:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ''
                
                # Extract content based on selector or default
                if selector:
                    content_elements = soup.select(selector)
                else:
                    # Common content selectors
                    content_elements = soup.find_all(['p', 'article', 'div.content', 'div.post-content'])
                
                content_text = '\n'.join([elem.get_text().strip() for elem in content_elements if elem.get_text().strip()])
                
                # Extract metadata
                meta_description = soup.find('meta', attrs={'name': 'description'})
                description = meta_description.get('content', '') if meta_description else ''
                
                scraped_data = {
                    'id': f"scraped_{hash(url)}",
                    'url': url,
                    'title': title_text,
                    'text': content_text,
                    'description': description,
                    'scraped_at': datetime.now(),
                    'source': 'web_scraping',
                    'domain': urlparse(url).netloc
                }
                
                ingested_data.append(scraped_data)
                logger.info(f"Successfully scraped content from {url}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                continue
        
        logger.info(f"Successfully scraped {len(ingested_data)} web pages")
        return ingested_data
    
    def ingest_from_csv(self, file_path: str, text_column: str = 'text', 
                       additional_columns: List[str] = None) -> List[Dict[str, Any]]:
        """Ingest data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV")
            
            ingested_data = []
            
            for index, row in df.iterrows():
                data_item = {
                    'id': f"csv_{index}",
                    'text': str(row[text_column]),
                    'source': 'csv',
                    'file_path': file_path,
                    'row_index': index
                }
                
                # Add additional columns if specified
                if additional_columns:
                    for col in additional_columns:
                        if col in df.columns:
                            data_item[col] = row[col]
                
                ingested_data.append(data_item)
            
            logger.info(f"Successfully ingested {len(ingested_data)} rows from CSV file {file_path}")
            return ingested_data
            
        except Exception as e:
            logger.error(f"CSV ingestion failed: {str(e)}")
            return []
    
    def process_ingested_data(self, data: List[Dict[str, Any]], model) -> List[Dict[str, Any]]:
        """Process ingested data with hate speech detection model"""
        processed_data = []
        
        for item in data:
            try:
                text = item.get('text', '')
                if not text.strip():
                    continue
                
                # Make prediction
                prediction_prob = model.predict([text])[0][0]
                prediction = "Hate Speech" if prediction_prob >= 0.5 else "No Hate Speech"
                
                # Add prediction results
                item.update({
                    'prediction': prediction,
                    'confidence': float(prediction_prob),
                    'processed_at': datetime.now(),
                    'text_length': len(text),
                    'word_count': len(text.split())
                })
                
                processed_data.append(item)
                
            except Exception as e:
                logger.error(f"Failed to process item {item.get('id', 'unknown')}: {str(e)}")
                item.update({
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'error': str(e),
                    'processed_at': datetime.now()
                })
                processed_data.append(item)
        
        logger.info(f"Processed {len(processed_data)} items with hate speech detection")
        return processed_data
    
    def export_processed_data(self, data: List[Dict[str, Any]], output_path: str, format: str = 'csv'):
        """Export processed data to file"""
        try:
            if format.lower() == 'csv':
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Successfully exported {len(data)} items to {output_path}")
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")


# Global instance
data_manager = DataIngestionManager()
