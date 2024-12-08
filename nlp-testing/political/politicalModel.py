from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flair.models import TextClassifier
from flair.data import Sentence
import matplotlib.pyplot as plt
import pandas as pd

# Load Flair sentiment model
flair_classifier = TextClassifier.load('en-sentiment')

# load model & tokenizer
model_name = "bucketresearch/politicalBiasBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# define labels
labels = {0: "Left", 1: "Center", 2: "Right"}

def split_into_chunks(text, max_length=512):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def classify_long_article(text):
    # Split into chunks
    chunks = split_into_chunks(text, max_length=512)
    
    # Classify each chunk
    chunk_results = [predict_political_bias(chunk) for chunk in chunks]
    
    # Aggregate results
    bias_counts = {"Left": 0, "Center": 0, "Right": 0}
    for result in chunk_results:
        bias_counts[result["bias"]] += 1
    
    # Determine overall bias
    overall_bias = max(bias_counts, key=bias_counts.get)
    
    return {"overall_bias": overall_bias, "chunk_results": chunk_results}

def predict_political_bias(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get model predictions
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs).item()
    
    # Return label and confidence score
    return {"bias": labels[prediction], "confidence": probs[0][prediction].item()}

# testing of the predict political_bias function
#article = "The government must act on climate change immediately to save our planet."
#result = predict_political_bias(article)
#print(result)  # Output: {'bias': 'Left', 'confidence': 0.87}

def flair_subjectivity(text):
    sentence = Sentence(text)
    flair_classifier.predict(sentence)
    score = sentence.labels[0].score
    return "Subjective" if score > 0.5 else "Objective"

def classify_article(text):
    bias_result = predict_political_bias(text)
    subjectivity = flair_subjectivity(text)
    return {
        "bias": bias_result["bias"],
        "bias_confidence": bias_result["confidence"],
        "subjectivity": subjectivity
    }

def read_article(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def sentiment_analysis(data):
    # Create a DataFrame from the list
    df = pd.DataFrame({'tokens': [line.strip() for line in data]})

    # Load the pre-trained sentiment analysis model
    classifier = TextClassifier.load('en-sentiment')

    # Iterate over each row in the DataFrame # :( its really time consuming :(
    for index, row in df.iterrows():
        text = row['tokens']  
        sentence = Sentence(text)
        
        # Predict the sentiment for the current sentence
        classifier.predict(sentence)
        
        # Get the predicted sentiment and score
        sentiment = sentence.labels[0].value
        score = sentence.labels[0].score
        
        # Update the DataFrame with the sentiment and score
        df.loc[index, 'sentiment'] = sentiment
        df.loc[index, 'score'] = score

    # Print the 'sentiment' column of the DataFrame
    print(df['sentiment'])
    # Count the occurrences of each sentiment label
    sentiment_counts = df['sentiment'].value_counts()

    # Plot a pie chart
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('Sentiment Analysis Results')

    plt.show()

    # Print the number of counts for each sentiment
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count}")

# test # 1 - syria article from cnn
#file_path = "syriaArticle.txt"
# Read the article
#article = read_article(file_path)
#data = [line for line in article.splitlines() if line.strip()]
#result = classify_article(article)
#print(result)
#sentiment_analysis(data)

# test #2 - kash FBI article from nyt
# seems to align (even though nyt is considered traditionally left-leaning, in the case of this article, right learning is apt)
#file_path = "kashFBIArticle.txt"
# Read the article
#article = read_article(file_path)
#data = [line for line in article.splitlines() if line.strip()]
#result = classify_article(article)
#print(result)
#entiment_analysis(data)

# test #3 - troudeau tarriff article from AP (considered libreral news source)
file_path = "trodeauArticle.txt"
# Read the article
article = read_article(file_path)
data = [line for line in article.splitlines() if line.strip()]
result = classify_article(article)
print(result)
sentiment_analysis(data)