import praw
import pandas as pd
import sqlite3
import schedule
import time
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 🔹 Reddit API Credentials (Replace with yours)
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "script:reddit_scraper:v1.0 (by u/your_reddit_username)"

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Connect to SQLite database
conn = sqlite3.connect("reddit_scraper.db")
cursor = conn.cursor()

# 🔹 Create Table for Storage
cursor.execute("""
CREATE TABLE IF NOT EXISTS reddit_posts (
    id TEXT PRIMARY KEY,
    subreddit TEXT,
    title TEXT,
    upvotes INTEGER,
    comments INTEGER,
    sentiment TEXT,
    predicted_topic TEXT,
    url TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def get_sentiment(text):
    """Analyze sentiment using VADER and return Positive, Neutral, or Negative."""
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def train_topic_classifier():
    """Train an ML model to classify topics of Reddit posts."""
    # Sample training data
    data = {
        "text": [
            "Bitcoin price reaches all-time high!",
            "Apple releases new M3 MacBook Pro",
            "Elections 2024: What to expect?",
            "Tesla's new self-driving feature is amazing",
            "Federal Reserve raises interest rates",
            "NASA discovers water on Mars",
            "Python 3.11 improves performance dramatically"
        ],
        "category": ["Finance", "Technology", "Politics", "Technology", "Finance", "Science", "Technology"]
    }

    df = pd.DataFrame(data)

    # Train ML Model
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    return pipeline

# Train model
topic_classifier = train_topic_classifier()

def predict_topic(text):
    """Predict topic category using ML model."""
    return topic_classifier.predict([text])[0]

def scrape_reddit(subreddit_name, post_type="hot", limit=10):
    """Scrapes subreddit posts, classifies topics, saves to SQLite & CSV."""
    try:
        subreddit = reddit.subreddit(subreddit_name)

        # Select post type
        if post_type == "hot":
            posts = subreddit.hot(limit=limit)
        elif post_type == "top":
            posts = subreddit.top(limit=limit)
        elif post_type == "new":
            posts = subreddit.new(limit=limit)
        else:
            print("Invalid post type! Choose 'hot', 'top', or 'new'.")
            return

        post_data = []
        for post in posts:
            sentiment = get_sentiment(post.title)  # Perform sentiment analysis
            predicted_topic = predict_topic(post.title)  # Predict topic

            post_data.append({
                "ID": post.id,
                "Subreddit": subreddit_name,
                "Title": post.title,
                "Upvotes": post.score,
                "Comments": post.num_comments,
                "Sentiment": sentiment,
                "Predicted Topic": predicted_topic,
                "Post URL": f"https://www.reddit.com{post.permalink}"
            })

            # Insert into SQLite Database
            cursor.execute("""
                INSERT OR IGNORE INTO reddit_posts 
                (id, subreddit, title, upvotes, comments, sentiment, predicted_topic, url) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (post.id, subreddit_name, post.title, post.score, post.num_comments, sentiment, predicted_topic, f"https://www.reddit.com{post.permalink}"))

        # Commit changes to DB
        conn.commit()

        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(post_data)
        csv_filename = f"{subreddit_name}_{post_type}_posts.csv"
        df.to_csv(csv_filename, index=False)

        print(f"Successfully scraped {len(post_data)} posts from r/{subreddit_name}")
        print(f"Data saved in {csv_filename} and SQLite database")

    except Exception as e:
        print(f"Error: {e}")

def scrape_comments(post_id, limit=5):
    """Scrapes top comments from a post and returns them."""
    try:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Load all comments

        comments_data = []
        for comment in submission.comments[:limit]:
            comments_data.append({
                "Comment": comment.body,
                "Upvotes": comment.score,
                "Sentiment": get_sentiment(comment.body)
            })

        return comments_data

    except Exception as e:
        print(f"❌ Error fetching comments: {e}")
        return []

def scheduled_scraper():
    """Automates scraping at regular intervals."""
    subreddit_input = "Python"  # Change to any subreddit you want
    scrape_reddit(subreddit_input, "hot", 10)

# 🔹 User Input for Scraping
subreddit_input = input("Enter subreddit name (without r/): ")
post_type_input = input("Choose post type (hot/top/new): ")
limit_input = int(input("How many posts? "))

# 🔹 Run scraper
scrape_reddit(subreddit_input, post_type_input, limit_input)

# 🔹 Run scheduling (e.g., scrape every 24 hours)
schedule.every(24).hours.do(scheduled_scraper)

while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
