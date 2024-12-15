from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import matplotlib.pyplot as plt
import requests
from newsdataapi import NewsDataApiClient


app = Flask(__name__)

api = NewsDataApiClient(apikey='pub_61575273e17c5f127cfdd57431a8f1ab3b599') 
NEWS_API_URL = "https://newsdata.io/api/1/news?apikey=pub_61575273e17c5f127cfdd57431a8f1ab3b599&q=technology "


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']  
    blob = TextBlob(text)       
    sentiment = blob.sentiment.polarity  
    subjectivity=blob.sentiment.subjectivity 


    if sentiment > 0:
        result = 'Positive' 
    elif sentiment < 0:
        result = 'Negative'
    else:
        result = 'Neutral'

    return jsonify({'sentiment': result, 'polarity': sentiment, 'subjectivity':subjectivity})


@app.route('/news', methods=['GET'])
def get_news():
    
    response = api.news_api(q='',language='en')  

    if response and 'results' in response:
        articles = response['results']
        news_items = []
      
        for article in articles[:3]:  
            description = article.get('description', '')
            blob = TextBlob(description)
            sentiment = blob.sentiment.polarity  
            subjectivity = blob.sentiment.subjectivity  
            
            if sentiment > 0:
                sentiment_label = 'Positive'
            elif sentiment < 0:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
            
            news_items.append({
                'title': article['title'],
                'link': article['link'],
                'sentiment': sentiment_label,
                'polarity': sentiment,
                'subjectivity': subjectivity
            })

       
        return jsonify(news_items)

if __name__ == "__main__":
    app.run(debug=True)