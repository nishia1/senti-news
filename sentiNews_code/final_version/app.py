from flask import Flask, render_template, request, jsonify
import requests
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flair.models import TextClassifier
from flair.data import Sentence
import torch

app = Flask(__name__)

# initialize the flair sentiment classifier
flair_classifier = TextClassifier.load('en-sentiment')

# initialize political bias detection model (politicalbiasBERT)
model_name = "bucketresearch/politicalBiasBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = {0: "left", 1: "center", 2: "right"}

# download necessary nltk data (just in case it's not downloaded)
nltk.download('punkt')

# your newsapi key - remember, don't share it in public!
API_KEY = "caa85a9e10804fe6904d4325ad667d2b"
url = "https://newsapi.org/v2/everything"

# function to fetch articles from newsapi
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
            # Skip articles that have "[Removed]" or any unwanted text
            if "[Removed]" in article.get("title", "") or "[Removed]" in article.get("content", ""):
                continue  # Skip this article if it has "[Removed]"
            articles.append({
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "source": article.get("source", {}).get("name", "Unknown")
            })
        return articles
    else:
        print(f"Failed to fetch articles: {response.status_code}")
        return []

# function to split text into sentences (just a little helper for analysis)
def split_sentences(text):
    return nltk.sent_tokenize(text)

# function for sentiment analysis using flair
def flair_subjectivity(text):
    sentence = Sentence(text)
    flair_classifier.predict(sentence)
    score = sentence.labels[0].score
    return "subjective" if score > 0.5 else "objective"

# function for political bias detection (using politicalBiasBERT)
def political_bias(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return labels[predicted_class]

# function to analyze sentiment and political bias for each sentence in the text
def sentiment_analysis_per_sentence(text):
    sentences = split_sentences(text)
    sentiment_results = {"positive": 0, "negative": 0, "neutral": 0}
    political_bias_results = {"left": 0, "center": 0, "right": 0}
    
    for sentence in sentences:
        sentiment = flair_subjectivity(sentence)
        bias = political_bias(sentence)
        
        # tally sentiment
        if sentiment == "positive":
            sentiment_results["positive"] += 1
        elif sentiment == "negative":
            sentiment_results["negative"] += 1
        else:
            sentiment_results["neutral"] += 1
        
        # tally political bias
        if bias == "left":
            political_bias_results["left"] += 1
        elif bias == "center":
            political_bias_results["center"] += 1
        else:
            political_bias_results["right"] += 1
    
    return sentiment_results, political_bias_results

# route to render homepage
@app.route('/')
def home():
    return render_template('index.html')

# route to fetch articles from API
@app.route('/fetch_articles', methods=['GET'])
def fetch_articles_route():
    query = request.args.get('query', default='politics', type=str)
    articles = fetch_articles(API_KEY, query=query)
    return jsonify(articles)

# route to analyze the article's content for sentiment and bias
@app.route('/analyze_article', methods=['POST'])
def analyze_article_route():
    content = request.json.get('content', '')
    sentiment_results, political_bias_results = sentiment_analysis_per_sentence(content)
    return jsonify({
        'sentiment': sentiment_results,
        'bias': political_bias_results
    })

# run the app, and let the magic happen!
if __name__ == '__main__':
    app.run(debug=True)