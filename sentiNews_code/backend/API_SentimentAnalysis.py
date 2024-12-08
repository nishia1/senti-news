from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flair.models import TextClassifier
from flair.data import Sentence
import matplotlib.pyplot as plt
import requests
from newsapi import NewsApiClient
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
# why does nltk hate me.... girl!
nltk.data.path.append("C:/Users/nishi/nltk_data")

# nishi's api key (she's too lazy to make 
# environment-specific var for privacy, 
# its ok what is someone going to do 
# with my free NewsApi key)
# don't steal pls make ur account pls and ty
API_KEY = "caa85a9e10804fe6904d4325ad667d2b"

# API endpoint
url = "https://newsapi.org/v2/everything"

# function to fetch articles from NewsAPI based on the user's selected keyword
def fetch_articles(api_key, query="politics", language="en", num_articles=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "apikey": api_key,
        "q": query,
        "language": language,
        "pageSize": num_articles
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "source": article.get("source", {}).get("name", "Unknown")
            })
        return articles
    else:
        print(f"Failed to fetch articles: {response.status_code}")
        return []

# Load Flair sentiment model
flair_classifier = TextClassifier.load('en-sentiment')

# Load model & tokenizer for political bias detection
model_name = "bucketresearch/politicalBiasBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define labels for political bias
labels = {0: "Left", 1: "Center", 2: "Right"}

def split_sentences(text):
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)
    return sentences

def flair_subjectivity(text):
    sentence = Sentence(text)
    flair_classifier.predict(sentence)
    score = sentence.labels[0].score
    return "Subjective" if score > 0.5 else "Objective"

def political_bias(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return labels[predicted_class]

# Function to analyze sentiment for each sentence in the article
def sentiment_analysis_per_sentence(text):
    sentences = split_sentences(text)
    
    # List to hold the sentiment results
    sentiment_results = {"Positive": 0, "Negative": 0, "Neutral": 0}
    political_bias_results = {"Left": 0, "Center": 0, "Right": 0}
    
    # Analyze sentiment and political bias for each sentence
    for sentence in sentences:
        sentiment = flair_subjectivity(sentence)
        bias = political_bias(sentence)
        
        if sentiment == "Positive":
            sentiment_results["Positive"] += 1
        elif sentiment == "Negative":
            sentiment_results["Negative"] += 1
        else:
            sentiment_results["Neutral"] += 1
        
        if bias == "Left":
            political_bias_results["Left"] += 1
        elif bias == "Center":
            political_bias_results["Center"] += 1
        else:
            political_bias_results["Right"] += 1
    
    # Create pie chart for sentiment analysis
    sentiment_counts = sentiment_results
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%')
    plt.title('Sentiment Analysis Results per Sentence')
    plt.show()
    
    # Create pie chart for political bias analysis
    bias_counts = political_bias_results
    plt.pie(bias_counts.values(), labels=bias_counts.keys(), autopct='%1.1f%%')
    plt.title('Political Bias Analysis Results per Sentence')
    plt.show()
    
    return sentiment_results, political_bias_results

# Define available keywords for the user to choose from
available_keywords = ["politics", "technology", "sports", "economy", "health"]

# Prompt user to choose a keyword
print("Select a keyword to fetch articles for sentiment and political bias analysis:")
for i, keyword in enumerate(available_keywords, 1):
    print(f"{i}. {keyword}")

# Get the user's selection
user_choice = int(input("Enter the number of your choice: "))
selected_keyword = available_keywords[user_choice - 1]

print(f"Fetching articles on {selected_keyword}...")

# Fetch articles based on the user's choice
articles = fetch_articles(API_KEY, query=selected_keyword)

# Analyze and display results for each article
for article in articles:
    print("Title:", article['title'])
    print("Description:", article['content'][:200])  # Only show the first 200 chars for brevity
    print("Source:", article['source'])
    print("-" * 80)
    sentiment_results, political_bias_results = sentiment_analysis_per_sentence(article['content'])
    print(f"Sentiment Analysis Results: {sentiment_results}")
    print(f"Political Bias Analysis Results: {political_bias_results}")
    print("=" * 80)
