from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib
import matplotlib.pyplot as plt
import time
import os

# Download VADER data if not already present
nltk.download('vader_lexicon')

# Initialize Sentiment Intensity Analyzer (VADER)
sia = SentimentIntensityAnalyzer()

# Load Ticket Classifier
with open('ticket_classifier.pkl', 'rb') as f:
    ticket_classifier = joblib.load(f)

# Load dataset
df = pd.read_csv('processed_tickets.csv')

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# CSV file path
CSV_FILE_PATH = 'comments_sentiment.csv'

# Write header to the CSV file if it doesn't already exist
def create_csv():
    if not os.path.isfile(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, 'w') as f:
            f.write("comment,sentiment\n")

create_csv()

def predict_sentiment(text):
    """Predicts sentiment using VADER (SIA) only."""
    score = sia.polarity_scores(text)['compound']
    return "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"

def predict_category(text):
    """Predicts ticket category."""
    return ticket_classifier.predict([text])[0]

def store_in_csv(comment, sentiment):
    """Stores comment and sentiment in the CSV file."""
    with open(CSV_FILE_PATH, 'a') as f:
        f.write(f'"{comment}",{sentiment}\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles sentiment and category analysis."""
    text = request.form['text']
    sentiment = predict_sentiment(text)
    category = predict_category(text)
    
    # Store the comment and sentiment in the CSV file
    store_in_csv(text, sentiment)

    return jsonify({'sentiment': sentiment, 'category': category})

@app.route('/generate_charts')
def generate_charts():
    """Generates and saves category and sentiment distribution charts."""
    time.sleep(2)  # Introduce a delay for chart generation

    # Generate Category Distribution Chart
    category_counts = df['category'].value_counts()
    labels = category_counts.index.tolist()
    values = category_counts.values.tolist()

    plt.figure(figsize=(6, 4))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Ticket Category Distribution")
    category_chart_path = 'static/category_chart.png'
    plt.savefig(category_chart_path)
    plt.close()

    # Generate Sentiment Distribution Chart
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    plt.figure(figsize=(6, 4))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Sentiment Trends")
    sentiment_chart_path = 'static/sentiment_chart.png'
    plt.savefig(sentiment_chart_path)
    plt.close()

    return jsonify({
        'category_chart_url': '/' + category_chart_path,
        'sentiment_chart_url': '/' + sentiment_chart_path
    })

@app.route('/get_comments', methods=['GET'])
def get_comments():
    """Fetches all stored comments and sentiments from the CSV."""
    comments_data = []
    with open(CSV_FILE_PATH, 'r') as f:
        lines = f.readlines()[1:]  # Skip the header
        for line in lines:
            comment, sentiment = line.strip().split(',')
            comments_data.append({'comment': comment, 'sentiment': sentiment})
    return jsonify(comments_data)

if __name__ == '__main__':
    app.run(debug=True)
