# app.py
from flask import Flask, render_template, request, jsonify
import nltk
from collections import Counter
from nltk.util import ngrams
import pandas as pd
import re

app = Flask(__name__)


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

bigram_freq = Counter()
trigram_freq = Counter()


def process_csv(file_path):
    global bigram_freq, trigram_freq
    
    try:
        df = pd.read_csv(file_path)
        
        # Clean text
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        column_name = "Text"  # Adjust column name if necessary
        
        df[column_name] = df[column_name].apply(clean_text)
        
        # Tokenize and generate n-grams
        df["Tokens"] = df[column_name].apply(nltk.word_tokenize)
        
        bigrams = []
        trigrams = []
        
        for tokens in df["Tokens"]:
            bigrams.extend(list(ngrams(tokens, 2)))
            trigrams.extend(list(ngrams(tokens, 3)))
        
        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)
        
        print(f"CSV processed successfully: {len(df)} rows, {len(bigram_freq)} bigrams, {len(trigram_freq)} trigrams")
        return True
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return False


CSV_PATH = "/Users/atharvajoshi/Documents/Chat_Team_CaseStudy FINAL.csv"
csv_processed = process_csv(CSV_PATH)

@app.route('/')
def index():
    global csv_processed
    return render_template('index.html', csv_processed=csv_processed)

@app.route('/predict', methods=['POST'])
def predict():
    global bigram_freq, trigram_freq
    
    data = request.json
    input_text = data.get('text', '')
    
    # Clean and tokenize input
    clean_input = re.sub(r"[^a-zA-Z\s]", "", input_text.lower())
    words = nltk.word_tokenize(clean_input)
    
    suggestions = []
    
    if len(words) >= 2:
        # Use trigram model
        last_bigram = tuple(words[-2:])
        possible_trigrams = [trigram for trigram in trigram_freq if trigram[:2] == last_bigram]
        predictions = sorted(possible_trigrams, key=lambda x: trigram_freq[x], reverse=True)
        suggestions = [word[-1] for word in predictions[:5]]  # Return top 5 suggestions
    elif len(words) == 1:
        # Use bigram model
        last_word = words[-1]
        possible_bigrams = [bigram for bigram in bigram_freq if bigram[0] == last_word]
        predictions = sorted(possible_bigrams, key=lambda x: bigram_freq[x], reverse=True)
        suggestions = [word[-1] for word in predictions[:5]]  # Return top 5 suggestions
    
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)